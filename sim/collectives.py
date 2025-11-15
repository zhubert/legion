"""
Collective Communication Operations for Distributed Training

Implements all-gather, reduce-scatter, and other collective operations
needed for ZeRO-3 style training. This simulation version uses shared
memory and multiprocessing primitives.
"""

import time
from typing import List, Dict, Optional, Callable
import torch
import torch.multiprocessing as mp


class CollectiveOps:
    """
    Simulated collective operations for distributed training.

    In a real distributed system, these would use NCCL, Gloo, or MPI.
    For our PoC, we simulate using shared memory and multiprocessing.
    """

    def __init__(self,
                 rank: int,
                 world_size: int,
                 backend: str = "shared_memory",
                 latency_ms: float = 0.0):
        """
        Args:
            rank: This worker's rank (0 to world_size-1)
            world_size: Total number of workers
            backend: Communication backend ('shared_memory' for simulation)
            latency_ms: Simulated network latency in milliseconds
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.latency_ms = latency_ms

        # Shared tensors for communication (set by coordinator)
        self.shared_tensors = None
        self.barriers = None

    def _simulate_latency(self):
        """Simulate network latency"""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

    def all_gather(self,
                   tensor: torch.Tensor,
                   async_op: bool = False) -> torch.Tensor:
        """
        All-gather: Each worker contributes a tensor, all receive concatenated result.

        In ZeRO-3, this is used to gather parameters from all workers before
        forward/backward pass.

        Args:
            tensor: Local tensor to contribute (shape: [local_size, ...])
            async_op: Whether to perform operation asynchronously

        Returns:
            Gathered tensor (shape: [local_size * world_size, ...])
        """
        if self.shared_tensors is None:
            raise RuntimeError("Shared tensors not initialized. Use set_shared_storage().")

        # Simulate network latency
        self._simulate_latency()

        # Write local tensor to shared storage
        self.shared_tensors[self.rank].copy_(tensor)

        # Wait for all workers to write
        if self.barriers:
            self.barriers['all_gather_write'].wait()

        # Simulate network latency for reading
        self._simulate_latency()

        # Gather all tensors
        gathered = torch.cat([self.shared_tensors[i].clone() for i in range(self.world_size)])

        # Synchronize after reading
        if self.barriers:
            self.barriers['all_gather_read'].wait()

        return gathered

    def reduce_scatter(self,
                      tensor: torch.Tensor,
                      op: str = "sum",
                      async_op: bool = False) -> torch.Tensor:
        """
        Reduce-scatter: Reduce tensor across all workers, scatter result.

        In ZeRO-3, this is used to aggregate gradients and send each worker
        its portion (for the parameters it owns).

        Args:
            tensor: Full tensor to reduce (shape: [total_size, ...])
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            async_op: Whether to perform operation asynchronously

        Returns:
            This worker's portion of reduced tensor (shape: [local_size, ...])
        """
        if self.shared_tensors is None:
            raise RuntimeError("Shared tensors not initialized. Use set_shared_storage().")

        # Simulate network latency
        self._simulate_latency()

        # Calculate chunk sizes (handle uneven division)
        total_size = tensor.shape[0]
        chunk_size = total_size // self.world_size
        remainder = total_size % self.world_size

        # Determine start/end for this rank's chunk
        if self.rank < remainder:
            start = self.rank * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = self.rank * chunk_size + remainder
            end = start + chunk_size

        # Extract this rank's chunk
        my_chunk = tensor[start:end]

        # Write chunk to shared storage
        self.shared_tensors[self.rank].copy_(my_chunk)

        # Wait for all workers to write
        if self.barriers:
            self.barriers['reduce_scatter_write'].wait()

        # Simulate network latency for reduction
        self._simulate_latency()

        # Reduce: sum contributions from all workers
        # Each worker reduces its own chunk independently
        reduced = torch.zeros_like(self.shared_tensors[self.rank])

        for i in range(self.world_size):
            # Get the i-th worker's contribution to this rank's chunk
            other_tensor = self.shared_tensors[i].clone()

            if op == "sum":
                reduced += other_tensor
            elif op == "mean":
                reduced += other_tensor / self.world_size
            elif op == "max":
                reduced = torch.maximum(reduced, other_tensor)
            elif op == "min":
                reduced = torch.minimum(reduced, other_tensor)
            else:
                raise ValueError(f"Unknown reduction op: {op}")

        # Synchronize after reduction
        if self.barriers:
            self.barriers['reduce_scatter_read'].wait()

        return reduced

    def broadcast(self,
                 tensor: torch.Tensor,
                 src: int = 0,
                 async_op: bool = False) -> torch.Tensor:
        """
        Broadcast: Send tensor from src to all workers.

        Args:
            tensor: Tensor to broadcast (only meaningful on src rank)
            src: Source rank to broadcast from
            async_op: Whether to perform operation asynchronously

        Returns:
            Broadcasted tensor
        """
        if self.shared_tensors is None:
            raise RuntimeError("Shared tensors not initialized. Use set_shared_storage().")

        # Simulate network latency
        self._simulate_latency()

        # Source writes tensor
        if self.rank == src:
            self.shared_tensors[0].copy_(tensor)

        # Wait for source to write
        if self.barriers:
            self.barriers['broadcast_write'].wait()

        # Simulate network latency for reading
        self._simulate_latency()

        # All workers read
        result = self.shared_tensors[0].clone()

        # Synchronize after reading
        if self.barriers:
            self.barriers['broadcast_read'].wait()

        return result

    def all_reduce(self,
                  tensor: torch.Tensor,
                  op: str = "sum",
                  async_op: bool = False) -> torch.Tensor:
        """
        All-reduce: Reduce tensor across all workers, all receive result.

        Args:
            tensor: Local tensor to reduce
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            async_op: Whether to perform operation asynchronously

        Returns:
            Reduced tensor (same shape as input)
        """
        if self.shared_tensors is None:
            raise RuntimeError("Shared tensors not initialized. Use set_shared_storage().")

        # Simulate network latency
        self._simulate_latency()

        # Write local tensor
        self.shared_tensors[self.rank].copy_(tensor)

        # Wait for all workers to write
        if self.barriers:
            self.barriers['all_reduce_write'].wait()

        # Simulate network latency for reduction
        self._simulate_latency()

        # Reduce
        result = torch.zeros_like(tensor)

        for i in range(self.world_size):
            other = self.shared_tensors[i].clone()

            if op == "sum":
                result += other
            elif op == "mean":
                result += other / self.world_size
            elif op == "max":
                result = torch.maximum(result, other)
            elif op == "min":
                result = torch.minimum(result, other)

        # Synchronize after reduction
        if self.barriers:
            self.barriers['all_reduce_read'].wait()

        return result

    def set_shared_storage(self, shared_tensors: List[torch.Tensor],
                          barriers: Dict[str, mp.Barrier]):
        """
        Set shared storage for communication.

        This is called by the coordinator to provide shared memory tensors
        and synchronization barriers.

        Args:
            shared_tensors: List of shared tensors (one per worker)
            barriers: Dictionary of barriers for synchronization
        """
        self.shared_tensors = shared_tensors
        self.barriers = barriers


class CollectiveCoordinator:
    """
    Coordinates collective operations across workers.

    Creates shared memory tensors and barriers for synchronization.
    """

    def __init__(self, world_size: int, max_tensor_size: int):
        """
        Args:
            world_size: Number of workers
            max_tensor_size: Maximum size of tensors to communicate
        """
        self.world_size = world_size
        self.max_tensor_size = max_tensor_size

        # Create shared tensors (one per worker)
        self.shared_tensors = [
            torch.zeros(max_tensor_size).share_memory_()
            for _ in range(world_size)
        ]

        # Create barriers for synchronization
        self.barriers = {
            'all_gather_write': mp.Barrier(world_size),
            'all_gather_read': mp.Barrier(world_size),
            'reduce_scatter_write': mp.Barrier(world_size),
            'reduce_scatter_read': mp.Barrier(world_size),
            'broadcast_write': mp.Barrier(world_size),
            'broadcast_read': mp.Barrier(world_size),
            'all_reduce_write': mp.Barrier(world_size),
            'all_reduce_read': mp.Barrier(world_size),
        }

    def get_collective_ops(self, rank: int, latency_ms: float = 0.0) -> CollectiveOps:
        """
        Create CollectiveOps instance for a worker.

        Args:
            rank: Worker rank
            latency_ms: Simulated network latency

        Returns:
            CollectiveOps instance configured for this worker
        """
        ops = CollectiveOps(rank, self.world_size, latency_ms=latency_ms)
        ops.set_shared_storage(self.shared_tensors, self.barriers)
        return ops


if __name__ == "__main__":
    print("Testing Collective Operations\n")

    world_size = 4
    max_size = 1000

    # Create coordinator
    coordinator = CollectiveCoordinator(world_size, max_size)

    # Test all-gather
    print("Testing all-gather:")
    tensors = [torch.ones(10) * i for i in range(world_size)]

    def test_all_gather(rank):
        ops = coordinator.get_collective_ops(rank)
        result = ops.all_gather(tensors[rank])
        expected = torch.cat([torch.ones(10) * i for i in range(world_size)])
        assert torch.allclose(result, expected), f"All-gather failed for rank {rank}"
        print(f"  Rank {rank}: ✓")

    # Run in sequence (multiprocessing test would be more complex)
    for rank in range(world_size):
        coordinator = CollectiveCoordinator(world_size, max_size)  # Reset barriers
        test_all_gather(rank)

    print("\nTesting reduce-scatter:")

    def test_reduce_scatter(rank):
        ops = coordinator.get_collective_ops(rank)
        # Each worker has full tensor
        full_tensor = torch.arange(40, dtype=torch.float32)
        result = ops.reduce_scatter(full_tensor, op="sum")

        # Expected: sum of 4 copies, then scatter
        chunk_size = 10
        start = rank * chunk_size
        end = start + chunk_size
        expected = torch.arange(40, dtype=torch.float32)[start:end] * world_size

        assert torch.allclose(result, expected), f"Reduce-scatter failed for rank {rank}"
        print(f"  Rank {rank}: ✓ (got chunk [{start}:{end}])")

    for rank in range(world_size):
        coordinator = CollectiveCoordinator(world_size, max_size)  # Reset barriers
        test_reduce_scatter(rank)

    print("\n✓ All collective operation tests passed!")
