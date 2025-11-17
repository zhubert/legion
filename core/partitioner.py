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
        Partition parameters across workers.

        Strategy: Simple round-robin by parameter count.
        Each worker gets roughly equal number of parameters.

        For a more sophisticated approach, we could:
        - Balance by memory size (considering optimizer states)
        - Group related parameters (e.g., keep layer together)
        - Consider computation graph dependencies
        """
        params_per_worker = self.total_params // self.world_size

        current_rank = 0
        current_count = 0
        current_start = 0
        current_params = []

        global_idx = 0

        for param in self.param_info:
            param_size = param['numel']

            # Check if adding this param would exceed allocation
            if (current_count + param_size > params_per_worker and
                current_rank < self.world_size - 1):
                # Finalize current partition
                partition = ParameterPartition(
                    rank=current_rank,
                    world_size=self.world_size,
                    param_names=current_params.copy(),
                    start_idx=current_start,
                    end_idx=global_idx
                )
                self.partitions.append(partition)

                # Start next partition
                current_rank += 1
                current_start = global_idx
                current_count = 0
                current_params = []

            # Add parameter to current partition
            current_params.append(param['name'])
            current_count += param_size
            global_idx += param_size

        # Add final partition
        if current_params:
            partition = ParameterPartition(
                rank=current_rank,
                world_size=self.world_size,
                param_names=current_params,
                start_idx=current_start,
                end_idx=global_idx
            )
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
