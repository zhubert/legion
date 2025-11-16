"""
Ring-based collective operations for bandwidth-efficient distributed training.

Implements ring all-reduce, ring all-gather, and ring reduce-scatter patterns
that minimize bandwidth usage by having each worker communicate with only
two neighbors in the ring topology.

Reference:
- Baidu's ring all-reduce: https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
"""

import asyncio
import logging
from typing import List, Optional
import torch

from communication.grpc_client import WorkerGRPCClient


logger = logging.getLogger(__name__)


class RingCollectiveOps:
    """
    Ring-based collective operations using gRPC.

    In a ring topology, each worker only communicates with its left and right
    neighbors, achieving O(1) bandwidth usage per worker instead of O(N).

    Ring topology for N workers:
        0 <-> 1 <-> 2 <-> ... <-> N-1 <-> 0
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        worker_id: str,
        worker_addresses: List[str],
        grpc_client: WorkerGRPCClient,
        timeout: float = 30.0
    ):
        """
        Initialize ring-based collective operations.

        Args:
            rank: This worker's rank (0 to world_size-1)
            world_size: Total number of workers
            worker_id: This worker's unique identifier
            worker_addresses: List of worker addresses in rank order
            grpc_client: gRPC client for making requests
            timeout: Timeout for collective operations in seconds
        """
        self.rank = rank
        self.world_size = world_size
        self.worker_id = worker_id
        self.worker_addresses = worker_addresses
        self.grpc_client = grpc_client
        self.timeout = timeout

        # Ring topology
        self.left_neighbor = (rank - 1) % world_size
        self.right_neighbor = (rank + 1) % world_size

        if len(worker_addresses) != world_size:
            raise ValueError(
                f"Number of worker addresses ({len(worker_addresses)}) "
                f"does not match world_size ({world_size})"
            )

        logger.info(
            f"Initialized RingCollectiveOps: rank={rank}, world_size={world_size}, "
            f"neighbors: left={self.left_neighbor}, right={self.right_neighbor}"
        )

    async def ring_all_reduce_async(
        self,
        tensor: torch.Tensor,
        op: str = "sum"
    ) -> torch.Tensor:
        """
        Ring all-reduce: reduce tensor across all workers using ring topology.

        Algorithm (simplified):
        1. Divide tensor into world_size chunks
        2. Reduce-scatter phase: Each worker sends its chunk to right neighbor,
           receives from left, reduces, and repeats for world_size-1 steps
        3. All-gather phase: Each worker sends reduced chunk to right neighbor,
           receives from left, and repeats for world_size-1 steps

        This achieves O(N) communication steps but with O(1) bandwidth per worker.

        Args:
            tensor: Local tensor to reduce
            op: Reduction operation ('sum', 'mean')

        Returns:
            Reduced tensor (same across all workers)
        """
        if self.world_size == 1:
            return tensor.clone()

        logger.debug(f"Ring all-reduce: rank {self.rank}, tensor shape {tensor.shape}")

        # Split tensor into chunks
        chunk_size = tensor.numel() // self.world_size
        remainder = tensor.numel() % self.world_size

        chunks = []
        offset = 0
        for i in range(self.world_size):
            size = chunk_size + (1 if i < remainder else 0)
            chunk = tensor.view(-1)[offset:offset + size].clone()
            chunks.append(chunk)
            offset += size

        # Phase 1: Reduce-scatter
        # After world_size-1 steps, each worker has one fully reduced chunk
        send_chunk_idx = self.rank
        recv_chunk_idx = self.left_neighbor

        for step in range(self.world_size - 1):
            # Send chunk to right neighbor
            send_task = self._send_chunk_to_neighbor(
                chunks[send_chunk_idx],
                self.right_neighbor,
                f"reduce_scatter_step_{step}"
            )

            # Receive chunk from left neighbor
            recv_task = self._receive_chunk_from_neighbor(
                self.left_neighbor,
                f"reduce_scatter_step_{step}"
            )

            # Wait for both operations
            received_chunk, send_success = await asyncio.gather(recv_task, send_task)

            # Reduce received chunk into our buffer
            if received_chunk is not None:
                if op == "sum":
                    chunks[recv_chunk_idx] += received_chunk
                elif op == "mean":
                    chunks[recv_chunk_idx] += received_chunk / self.world_size

            # Update indices for next iteration
            send_chunk_idx = recv_chunk_idx
            recv_chunk_idx = (recv_chunk_idx - 1) % self.world_size

        # Phase 2: All-gather
        # After world_size-1 steps, all workers have all reduced chunks
        send_chunk_idx = (self.rank + 1) % self.world_size

        for step in range(self.world_size - 1):
            # Send chunk to right neighbor
            send_task = self._send_chunk_to_neighbor(
                chunks[send_chunk_idx],
                self.right_neighbor,
                f"all_gather_step_{step}"
            )

            # Receive chunk from left neighbor
            recv_chunk_idx = (send_chunk_idx - 1) % self.world_size
            recv_task = self._receive_chunk_from_neighbor(
                self.left_neighbor,
                f"all_gather_step_{step}"
            )

            # Wait for both operations
            received_chunk, send_success = await asyncio.gather(recv_task, send_task)

            # Update our chunk with received data
            if received_chunk is not None:
                chunks[recv_chunk_idx] = received_chunk

            # Update index for next iteration
            send_chunk_idx = recv_chunk_idx

        # Concatenate all chunks to form final result
        result = torch.cat(chunks)

        # Reshape to original shape
        result = result.view(tensor.shape)

        logger.debug(f"Ring all-reduce complete: rank {self.rank}")

        return result

    async def ring_all_gather_async(
        self,
        tensor: torch.Tensor,
        parameter_name: str = "parameter"
    ) -> torch.Tensor:
        """
        Ring all-gather: gather tensors from all workers using ring topology.

        Each worker has a chunk of the full tensor. After world_size-1 steps,
        all workers have the complete tensor.

        Args:
            tensor: This worker's chunk
            parameter_name: Name for identification

        Returns:
            Concatenated tensor from all workers
        """
        if self.world_size == 1:
            return tensor.clone()

        logger.debug(
            f"Ring all-gather: rank {self.rank}, chunk shape {tensor.shape}"
        )

        # Initialize buffer with our chunk at the correct position
        chunks = [None] * self.world_size
        chunks[self.rank] = tensor.clone()

        # Ring all-gather: send our chunk around the ring
        send_chunk_idx = self.rank
        recv_chunk_idx = self.left_neighbor

        for step in range(self.world_size - 1):
            # Send chunk to right neighbor
            send_task = self._send_chunk_to_neighbor(
                chunks[send_chunk_idx],
                self.right_neighbor,
                f"all_gather_{parameter_name}_step_{step}"
            )

            # Receive chunk from left neighbor
            recv_task = self._receive_chunk_from_neighbor(
                self.left_neighbor,
                f"all_gather_{parameter_name}_step_{step}"
            )

            # Wait for both operations
            received_chunk, send_success = await asyncio.gather(recv_task, send_task)

            # Store received chunk
            if received_chunk is not None:
                chunks[recv_chunk_idx] = received_chunk

            # Update indices
            send_chunk_idx = recv_chunk_idx
            recv_chunk_idx = (recv_chunk_idx - 1) % self.world_size

        # Concatenate all chunks
        result = torch.cat([c for c in chunks if c is not None])

        logger.debug(f"Ring all-gather complete: rank {self.rank}, result shape {result.shape}")

        return result

    async def ring_reduce_scatter_async(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
        step: int = 0
    ) -> torch.Tensor:
        """
        Ring reduce-scatter: reduce tensor and scatter to owners using ring topology.

        After world_size-1 steps, each worker has its portion of the reduced tensor.

        Args:
            tensor: Full tensor to reduce and scatter
            op: Reduction operation ('sum', 'mean')
            step: Training step number

        Returns:
            This worker's reduced chunk
        """
        if self.world_size == 1:
            return tensor.clone()

        logger.debug(f"Ring reduce-scatter: rank {self.rank}, tensor shape {tensor.shape}")

        # Split tensor into chunks
        chunk_size = tensor.shape[0] // self.world_size
        remainder = tensor.shape[0] % self.world_size

        chunks = []
        offset = 0
        for i in range(self.world_size):
            size = chunk_size + (1 if i < remainder else 0)
            chunk = tensor[offset:offset + size].clone()
            chunks.append(chunk)
            offset += size

        # Reduce-scatter phase
        send_chunk_idx = self.rank
        recv_chunk_idx = self.left_neighbor

        for step in range(self.world_size - 1):
            # Send chunk to right neighbor
            send_task = self._send_chunk_to_neighbor(
                chunks[send_chunk_idx],
                self.right_neighbor,
                f"reduce_scatter_step_{step}"
            )

            # Receive chunk from left neighbor
            recv_task = self._receive_chunk_from_neighbor(
                self.left_neighbor,
                f"reduce_scatter_step_{step}"
            )

            # Wait for both operations
            received_chunk, send_success = await asyncio.gather(recv_task, send_task)

            # Reduce received chunk
            if received_chunk is not None:
                if op == "sum":
                    chunks[recv_chunk_idx] += received_chunk
                elif op == "mean":
                    chunks[recv_chunk_idx] += received_chunk / self.world_size

            # Update indices
            send_chunk_idx = recv_chunk_idx
            recv_chunk_idx = (recv_chunk_idx - 1) % self.world_size

        # Return this worker's reduced chunk
        my_chunk = chunks[self.rank]

        logger.debug(f"Ring reduce-scatter complete: rank {self.rank}, chunk shape {my_chunk.shape}")

        return my_chunk

    async def _send_chunk_to_neighbor(
        self,
        chunk: torch.Tensor,
        neighbor_rank: int,
        chunk_id: str
    ) -> bool:
        """
        Send a tensor chunk to a neighbor in the ring.

        Args:
            chunk: Tensor chunk to send
            neighbor_rank: Rank of the neighbor
            chunk_id: Identifier for this chunk transfer

        Returns:
            True if successful, False otherwise
        """
        neighbor_address = self.worker_addresses[neighbor_rank]

        try:
            success = await self.grpc_client.send_gradients(
                worker_address=neighbor_address,
                gradients=chunk,
                step=0,  # Not used for ring operations
                shard_start=0,
                shard_end=-1
            )
            return success
        except Exception as e:
            logger.error(f"Error sending chunk to neighbor {neighbor_rank}: {e}")
            return False

    async def _receive_chunk_from_neighbor(
        self,
        neighbor_rank: int,
        chunk_id: str
    ) -> Optional[torch.Tensor]:
        """
        Receive a tensor chunk from a neighbor in the ring.

        Note: In a real implementation, this would use a proper message queue
        or synchronized communication pattern. For now, we simulate by
        fetching from the neighbor's parameter store.

        Args:
            neighbor_rank: Rank of the neighbor
            chunk_id: Identifier for this chunk transfer

        Returns:
            Received tensor chunk or None if failed
        """
        # TODO: Implement proper synchronous send/receive pattern
        # For now, this is a placeholder that would need to be integrated
        # with a message passing system

        logger.warning(
            "Ring collectives require synchronous message passing, "
            "which is not yet fully implemented in the gRPC layer"
        )

        return None


def compute_ring_communication_cost(world_size: int, tensor_size: int) -> dict:
    """
    Calculate communication cost for ring-based collectives.

    Args:
        world_size: Number of workers
        tensor_size: Size of tensor in bytes

    Returns:
        Dictionary with cost analysis
    """
    # Ring all-reduce: 2(N-1) communication steps, each transferring 1/N of data
    ring_steps = 2 * (world_size - 1)
    ring_data_per_step = tensor_size / world_size
    ring_total_bandwidth = ring_steps * ring_data_per_step

    # Naive all-reduce: Each worker sends to all others
    naive_total_bandwidth = world_size * (world_size - 1) * tensor_size

    return {
        'world_size': world_size,
        'tensor_size_bytes': tensor_size,
        'ring_steps': ring_steps,
        'ring_data_per_step_bytes': ring_data_per_step,
        'ring_total_bandwidth_bytes': ring_total_bandwidth,
        'naive_total_bandwidth_bytes': naive_total_bandwidth,
        'bandwidth_reduction_factor': naive_total_bandwidth / ring_total_bandwidth if ring_total_bandwidth > 0 else 0,
    }


if __name__ == "__main__":
    # Demonstrate bandwidth savings
    print("Ring Collective Bandwidth Analysis")
    print("=" * 60)

    # Example: 4 workers, 1GB model
    for world_size in [2, 4, 8, 16, 32]:
        tensor_size = 1 * 1024 * 1024 * 1024  # 1GB
        cost = compute_ring_communication_cost(world_size, tensor_size)

        print(f"\nWorld size: {world_size} workers")
        print(f"  Ring steps: {cost['ring_steps']}")
        print(f"  Ring data per step: {cost['ring_data_per_step_bytes'] / (1024**2):.1f} MB")
        print(f"  Ring total bandwidth: {cost['ring_total_bandwidth_bytes'] / (1024**3):.2f} GB")
        print(f"  Naive total bandwidth: {cost['naive_total_bandwidth_bytes'] / (1024**3):.2f} GB")
        print(f"  Bandwidth reduction: {cost['bandwidth_reduction_factor']:.1f}x")

    print("\n" + "=" * 60)
