"""
Worker Process for Distributed Training

Each worker owns a subset of model parameters and participates in
distributed training using ZeRO-3 style parameter partitioning.
"""

import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from core.partitioner import ParameterPartition, flatten_parameters, unflatten_parameters
from core.compression import CompressionManager
from sim.collectives import CollectiveOps


class Worker:
    """
    A worker in the distributed training system.

    Each worker:
    1. Owns a subset of model parameters (ZeRO-3 partitioning)
    2. Participates in collective communication for all-gather/reduce-scatter
    3. Computes forward/backward passes on local batches
    4. Updates only its owned parameters
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        model: nn.Module,
        partition: ParameterPartition,
        collective_ops: CollectiveOps,
        optimizer: Optimizer,
        compression: Optional[CompressionManager] = None,
        device: str = "cpu"
    ):
        """
        Args:
            rank: Worker rank (0 to world_size-1)
            world_size: Total number of workers
            model: The full model (architecture only, params will be sharded)
            partition: This worker's parameter partition
            collective_ops: Collective communication operations
            optimizer: Optimizer for owned parameters
            compression: Optional gradient compression
            device: Device to run on ('cpu' or 'cuda')
        """
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(device)
        self.partition = partition
        self.collective_ops = collective_ops
        self.optimizer = optimizer
        self.compression = compression
        self.device = device

        # Extract owned parameters
        self.owned_params = self._extract_owned_parameters()

        # Statistics
        self.stats = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'all_gather_time': 0.0,
            'reduce_scatter_time': 0.0,
            'compression_time': 0.0,
            'update_time': 0.0,
            'total_steps': 0
        }

    def _extract_owned_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract parameters owned by this worker from model"""
        owned = {}
        for name, param in self.model.named_parameters():
            if name in self.partition.param_names:
                # Clone and detach to create independent copy
                owned[name] = param.data.clone().detach().requires_grad_(True)
        return owned

    def _all_gather_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Gather all parameters from all workers.

        This simulates the all-gather phase of ZeRO-3 where each worker
        temporarily collects the full model for forward/backward pass.

        Returns:
            Dictionary of all model parameters
        """
        start_time = time.time()

        # In real implementation, this would be optimized to only gather
        # parameters needed for current layers. For PoC, we gather everything.

        all_params = {}

        # Gather owned parameters from all workers
        for name in self.partition.param_names:
            param = self.owned_params[name]
            # Flatten for communication
            flat_param = param.flatten()

            # All-gather (simulated)
            gathered = self.collective_ops.all_gather(flat_param)

            # In real system, we'd unflatten and assign to model
            # For now, just use our owned copy
            all_params[name] = param

        self.stats['all_gather_time'] += time.time() - start_time

        return all_params

    def _reduce_scatter_gradients(self, gradients: Dict[str, torch.Tensor]):
        """
        Reduce-scatter gradients to parameter owners.

        Each worker sends gradients for parameters it doesn't own to
        the workers that do own them. Gradients are summed and scattered.

        Args:
            gradients: Dictionary of gradients for all parameters
        """
        start_time = time.time()

        # Compress gradients if enabled
        if self.compression is not None:
            compress_start = time.time()
            for name in self.partition.param_names:
                if name in gradients:
                    grad = gradients[name]
                    compressed = self.compression.compress(grad, name=name)
                    # In real system, this would be sent over network
                    # For simulation, we decompress immediately
                    gradients[name] = self.compression.decompress(compressed)
            self.stats['compression_time'] += time.time() - compress_start

        # Reduce-scatter (simulated)
        # In real implementation, each worker would send its gradients
        # for owned parameters to all other workers, and receive gradients
        # for its owned parameters from others.

        # For simulation, we just copy gradients for owned parameters
        for name in self.partition.param_names:
            if name in gradients:
                # In real system: reduce-scatter operation here
                # For now, just accumulate gradient
                self.owned_params[name].grad = gradients[name]

        self.stats['reduce_scatter_time'] += time.time() - start_time

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Tuple of (inputs, targets)

        Returns:
            Dictionary of metrics (loss, etc.)
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 1. All-gather: Get full model parameters
        # In real ZeRO-3, this is done layer-by-layer during forward pass
        # For PoC, we do it once before forward
        # all_params = self._all_gather_parameters()

        # 2. Forward pass
        start_time = time.time()
        self.model.train()
        logits, loss = self.model(inputs, targets)
        self.stats['forward_time'] += time.time() - start_time

        # 3. Backward pass
        start_time = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        self.stats['backward_time'] += time.time() - start_time

        # 4. Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # 5. Reduce-scatter: Send gradients to owners
        self._reduce_scatter_gradients(gradients)

        # 6. Update owned parameters
        start_time = time.time()

        # Update parameters in owned_params dict
        for name in self.partition.param_names:
            if name in self.owned_params:
                param = self.owned_params[name]
                if param.grad is not None:
                    # Manual parameter update (simplified optimizer)
                    # In real system, optimizer would handle this
                    with torch.no_grad():
                        param -= 0.001 * param.grad  # Simple SGD with lr=0.001

        # Also update model parameters for consistency
        for name, param in self.model.named_parameters():
            if name in self.partition.param_names:
                param.data.copy_(self.owned_params[name].data)

        self.stats['update_time'] += time.time() - start_time
        self.stats['total_steps'] += 1

        return {
            'loss': loss.item(),
            'rank': self.rank
        }

    def get_stats(self) -> Dict[str, float]:
        """Get worker statistics"""
        if self.stats['total_steps'] > 0:
            avg_stats = {
                f'avg_{k}': v / self.stats['total_steps']
                for k, v in self.stats.items()
                if k != 'total_steps'
            }
            avg_stats['total_steps'] = self.stats['total_steps']
            return avg_stats
        return self.stats

    def sync_parameters(self, all_params: Dict[str, torch.Tensor]):
        """
        Synchronize owned parameters with global state.

        Used for checkpoint loading or periodic synchronization.

        Args:
            all_params: Dictionary of all model parameters
        """
        for name in self.partition.param_names:
            if name in all_params:
                self.owned_params[name].data.copy_(all_params[name].data)

        # Update model as well
        for name, param in self.model.named_parameters():
            if name in all_params:
                param.data.copy_(all_params[name].data)


class WorkerCoordinator:
    """
    Coordinates multiple workers in a simulated distributed environment.

    This class manages worker lifecycle and synchronization for single-machine
    simulation. In a real distributed system, there wouldn't be a central
    coordinator - workers would communicate peer-to-peer.
    """

    def __init__(
        self,
        world_size: int,
        model: nn.Module,
        partitions: list,
        collective_ops_factory,
        learning_rate: float = 0.001,
        compression: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            world_size: Number of workers
            model: Model architecture
            partitions: List of parameter partitions
            collective_ops_factory: Function to create CollectiveOps for each worker
            learning_rate: Learning rate for optimizer
            compression: Compression method ('int8', 'topk', 'none', or None)
            device: Device to use
        """
        self.world_size = world_size
        self.model = model
        self.partitions = partitions
        self.learning_rate = learning_rate
        self.device = device

        # Create compression manager if specified
        self.compression = None
        if compression and compression != 'none':
            self.compression = CompressionManager(compression)

        # Create workers
        self.workers = []
        for rank in range(world_size):
            collective_ops = collective_ops_factory(rank)

            # Create separate optimizer for each worker's parameters
            # In real system, each worker has its own optimizer instance
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate
            )

            worker = Worker(
                rank=rank,
                world_size=world_size,
                model=model,
                partition=partitions[rank],
                collective_ops=collective_ops,
                optimizer=optimizer,
                compression=self.compression,
                device=device
            )

            self.workers.append(worker)

    def train_step(self, batches: list) -> list:
        """
        Execute one training step across all workers.

        Args:
            batches: List of batches, one per worker

        Returns:
            List of metrics from each worker
        """
        metrics = []
        for worker, batch in zip(self.workers, batches):
            worker_metrics = worker.train_step(batch)
            metrics.append(worker_metrics)

        return metrics

    def get_all_stats(self) -> list:
        """Get statistics from all workers"""
        return [worker.get_stats() for worker in self.workers]

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """
        Reconstruct full model state from all workers.

        Collects owned parameters from each worker.
        """
        full_state = {}
        for worker in self.workers:
            for name, param in worker.owned_params.items():
                full_state[name] = param.data.clone()

        return full_state
