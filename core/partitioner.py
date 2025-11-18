"""
Parameter Partitioner for ZeRO-3 style sharding

This module handles partitioning model parameters across workers.
Each worker "owns" a subset of parameters and is responsible for
updating them during training.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class ParameterPartition:
    """Represents a partition of parameters owned by a single worker"""

    def __init__(self, rank: int, world_size: int, param_names: List[str],
                 start_idx: int, end_idx: int):
        """
        Args:
            rank: Worker rank (0 to world_size-1)
            world_size: Total number of workers
            param_names: List of parameter names in this partition
            start_idx: Starting parameter index in global flattened view
            end_idx: Ending parameter index (exclusive)
        """
        self.rank = rank
        self.world_size = world_size
        self.param_names = param_names
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_params = end_idx - start_idx

        # Map parameter names to their slices in this partition
        # Format: {param_name: (start_offset, end_offset, total_param_size)}
        # Example: {"embedding.weight": (10000, 15000, 128000)}
        # means this worker owns elements [10000:15000] of a 128000-element parameter
        self.param_slices: Dict[str, Tuple[int, int, int]] = {}

    def __repr__(self):
        return (f"ParameterPartition(rank={self.rank}, "
                f"params={len(self.param_names)}, "
                f"size={self.num_params})")


class Partitioner:
    """
    Partitions model parameters across workers using ZeRO-3 strategy.

    In ZeRO-3, each worker owns a subset of model parameters. During training:
    1. All-gather: Workers collect parameters they need for computation
    2. Compute: Forward and backward passes
    3. Reduce-scatter: Gradients are sent back to parameter owners
    4. Update: Each worker updates only its owned parameters
    """

    def __init__(self, model: nn.Module, world_size: int):
        """
        Args:
            model: The PyTorch model to partition
            world_size: Number of workers to partition across
        """
        self.model = model
        self.world_size = world_size
        self.partitions = []

        # Analyze model parameters
        self.param_info = self._analyze_parameters()
        self.total_params = sum(info['numel'] for info in self.param_info)

        # Create partitions
        self._create_partitions()

    def _analyze_parameters(self) -> List[Dict]:
        """Analyze model parameters and gather metadata"""
        param_info = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_info.append({
                    'name': name,
                    'shape': param.shape,
                    'numel': param.numel(),
                    'dtype': param.dtype
                })

        return param_info

    def _create_partitions(self):
        """
        Partition parameters across workers using flat splitting.

        Strategy: Flatten all parameters into one 1D space and split evenly.
        Each worker gets exactly total_params/world_size elements.
        Large parameters may be split across multiple workers.

        This enables ring collectives which require all workers to have
        same-sized tensors.
        """
        params_per_worker = self.total_params // self.world_size

        # Create partitions with even splits
        for rank in range(self.world_size):
            start_idx = rank * params_per_worker
            # Last worker gets any remainder
            if rank == self.world_size - 1:
                end_idx = self.total_params
            else:
                end_idx = start_idx + params_per_worker

            partition = ParameterPartition(
                rank=rank,
                world_size=self.world_size,
                param_names=[],
                start_idx=start_idx,
                end_idx=end_idx
            )

            # Map flat range to actual parameters
            global_offset = 0
            for param_info in self.param_info:
                param_name = param_info['name']
                param_size = param_info['numel']
                param_start = global_offset
                param_end = global_offset + param_size

                # Check if this parameter overlaps with this partition's range
                if param_end > start_idx and param_start < end_idx:
                    # Calculate overlap
                    overlap_start = max(param_start, start_idx) - param_start
                    overlap_end = min(param_end, end_idx) - param_start

                    # Store slice information
                    partition.param_slices[param_name] = (
                        overlap_start,  # Start offset within the parameter
                        overlap_end,    # End offset within the parameter
                        param_size      # Total parameter size
                    )

                    # Add to param_names if not already there
                    if param_name not in partition.param_names:
                        partition.param_names.append(param_name)

                global_offset += param_size

            self.partitions.append(partition)

    def get_partition(self, rank: int) -> ParameterPartition:
        """Get the partition for a specific worker rank"""
        if rank >= len(self.partitions):
            raise ValueError(f"Rank {rank} out of range (max: {len(self.partitions)-1})")
        return self.partitions[rank]

    def get_owned_parameters(self, rank: int) -> Dict[str, torch.Tensor]:
        """
        Extract the parameters owned by a specific worker.

        Returns:
            Dictionary mapping parameter names to their tensors
        """
        partition = self.get_partition(rank)
        owned_params = {}

        for name in partition.param_names:
            param = dict(self.model.named_parameters())[name]
            owned_params[name] = param.clone().detach()

        return owned_params

    def extract_flat_shard(self, rank: int) -> torch.Tensor:
        """
        Extract this rank's flat parameter shard from the model.

        Args:
            rank: Worker rank

        Returns:
            Flat 1D tensor containing this worker's parameter shard
        """
        partition = self.get_partition(rank)
        flat_pieces = []

        for param_name in sorted(partition.param_names):
            # Get the full parameter
            param = dict(self.model.named_parameters())[param_name]
            param_flat = param.data.flatten()

            # Extract only this worker's slice
            start, end, total = partition.param_slices[param_name]
            flat_pieces.append(param_flat[start:end])

        return torch.cat(flat_pieces)

    def reconstruct_from_flat_shards(self, flat_shards: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Reconstruct full parameters from flat shards collected via ring all-gather.

        Args:
            flat_shards: List of flat tensors, one per worker (in rank order)

        Returns:
            Dictionary mapping parameter names to full reconstructed tensors
        """
        # Concatenate all shards to get full flattened parameter space
        full_flat = torch.cat(flat_shards)

        # Reconstruct each parameter
        params = {}
        global_offset = 0

        for param_info in self.param_info:
            param_name = param_info['name']
            param_size = param_info['numel']
            param_shape = param_info['shape']

            # Extract this parameter's data from the flat space
            param_data = full_flat[global_offset:global_offset + param_size]
            params[param_name] = param_data.reshape(param_shape)

            global_offset += param_size

        return params

    def apply_flat_gradients_to_owned_params(self, rank: int, flat_grads: torch.Tensor):
        """
        Apply flat gradient shard to this worker's owned parameter slices.

        Args:
            rank: Worker rank
            flat_grads: Flat 1D tensor of reduced gradients for this worker's shard
        """
        partition = self.get_partition(rank)
        offset = 0

        for param_name in sorted(partition.param_names):
            param = dict(self.model.named_parameters())[param_name]
            start, end, total = partition.param_slices[param_name]
            slice_size = end - start

            # Extract gradient slice for this parameter
            grad_slice = flat_grads[offset:offset + slice_size]

            # Apply to the corresponding slice of the parameter's gradient
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)

            param_grad_flat = param.grad.data.flatten()
            param_grad_flat[start:end] = grad_slice.to(param.dtype)
            param.grad.data = param_grad_flat.reshape(param.shape)

            offset += slice_size

    def print_partition_info(self):
        """Print information about the partitioning"""
        print(f"\n{'='*60}")
        print(f"Parameter Partitioning Summary")
        print(f"{'='*60}")
        print(f"Total parameters: {self.total_params:,}")
        print(f"Number of workers: {self.world_size}")
        print(f"Average params per worker: {self.total_params // self.world_size:,}")
        print(f"\nPartition Details:")
        print(f"{'-'*60}")

        for partition in self.partitions:
            print(f"Rank {partition.rank}:")
            print(f"  Parameters: {partition.num_params:,} "
                  f"({partition.num_params / self.total_params * 100:.1f}%)")
            print(f"  Tensors: {len(partition.param_names)}")
            print(f"  Range: [{partition.start_idx:,}, {partition.end_idx:,})")
            if len(partition.param_names) <= 5:
                print(f"  Names: {partition.param_names}")
            else:
                print(f"  Names: {partition.param_names[:2]} ... {partition.param_names[-2:]}")
            print()


def flatten_parameters(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten a dictionary of parameters into a single 1D tensor.

    Useful for communication - easier to send one contiguous tensor.
    """
    flat_params = []
    for name in sorted(params.keys()):
        flat_params.append(params[name].flatten())
    return torch.cat(flat_params)


def unflatten_parameters(flat_tensor: torch.Tensor,
                        param_shapes: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
    """
    Unflatten a 1D tensor back into a dictionary of parameters.

    Args:
        flat_tensor: Flattened 1D tensor
        param_shapes: Dictionary mapping parameter names to their shapes

    Returns:
        Dictionary of reshaped parameters
    """
    params = {}
    offset = 0

    for name in sorted(param_shapes.keys()):
        shape = param_shapes[name]
        numel = torch.Size(shape).numel()
        params[name] = flat_tensor[offset:offset+numel].reshape(shape)
        offset += numel

    return params


if __name__ == "__main__":
    # Example usage
    from model import TinyGPT

    print("Testing Parameter Partitioner\n")

    # Create a tiny model
    model = TinyGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=256
    )

    # Partition across 4 workers
    partitioner = Partitioner(model, world_size=4)
    partitioner.print_partition_info()

    # Test getting owned parameters
    print("\nTesting parameter extraction for rank 0:")
    owned = partitioner.get_owned_parameters(rank=0)
    print(f"Extracted {len(owned)} parameter tensors")

    # Test flattening
    flat = flatten_parameters(owned)
    print(f"Flattened to tensor of shape: {flat.shape}")

    # Test unflattening
    param_shapes = {name: param.shape for name, param in owned.items()}
    unflat = unflatten_parameters(flat, param_shapes)
    print(f"Unflattened back to {len(unflat)} tensors")

    # Verify correctness
    for name in owned.keys():
        assert torch.allclose(owned[name], unflat[name]), f"Mismatch in {name}"
    print("✓ Flatten/unflatten test passed!")
