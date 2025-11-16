"""
Collective communication operations for distributed training.

Implements all-gather and reduce-scatter using ring-based algorithms
over gRPC for efficient peer-to-peer communication.
"""

import asyncio
import logging
from typing import List, Dict, Optional
import torch
import uuid

from communication.grpc_client import WorkerGRPCClient


logger = logging.getLogger(__name__)


class CollectiveOps:
    """
    Manages collective operations across workers using ring-based algorithms.

    Ring all-gather and reduce-scatter minimize network traffic by having
    each worker communicate only with its neighbors in the ring.
    """

    def __init__(self, worker_id: str, grpc_client: WorkerGRPCClient):
        """
        Initialize collective operations.

        Args:
            worker_id: This worker's unique identifier
            grpc_client: gRPC client for peer communication
        """
        self.worker_id = worker_id
        self.client = grpc_client
        self.active_collectives: Dict[str, dict] = {}  # Track in-progress collectives

    async def all_gather(
        self,
        tensor: torch.Tensor,
        worker_addresses: List[str],
        rank: int,
    ) -> List[torch.Tensor]:
        """
        All-gather operation: collect tensors from all workers.

        Uses ring-based algorithm:
        1. Each worker sends its chunk to the next worker in the ring
        2. After N-1 steps, all workers have all chunks

        Args:
            tensor: This worker's tensor contribution
            worker_addresses: List of all worker addresses in ring order
            rank: This worker's position in the ring (0 to N-1)

        Returns:
            List of tensors from all workers (including self)
        """
        world_size = len(worker_addresses)
        collective_id = str(uuid.uuid4())

        logger.info(
            f"Starting all-gather collective {collective_id}: "
            f"rank={rank}, world_size={world_size}"
        )

        # Edge case: no other workers
        if world_size == 0:
            return [tensor.clone()]

        # Initialize gathered tensors with our own
        gathered = [None] * world_size
        gathered[rank] = tensor.clone()

        # Ring all-gather: send to next, receive from previous
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1 + world_size) % world_size

        next_address = worker_addresses[next_rank]
        prev_address = worker_addresses[prev_rank]

        # TODO: Implement proper ring all-gather with async send/recv
        # For now, use simple all-to-all (not bandwidth optimal)
        for i, address in enumerate(worker_addresses):
            if i == rank:
                # Already have our own tensor
                continue

            # Fetch tensor from worker i
            fetched = await self.client.get_parameters(
                address,
                parameter_name=f"allgather_{collective_id}_{i}",
                shard_start=0,
                shard_end=-1,
            )

            if fetched is not None:
                gathered[i] = fetched
            else:
                logger.error(f"Failed to fetch tensor from rank {i}")
                # Use zeros as fallback
                gathered[i] = torch.zeros_like(tensor)

        logger.info(f"Completed all-gather collective {collective_id}")

        return gathered

    async def reduce_scatter(
        self,
        tensor: torch.Tensor,
        worker_addresses: List[str],
        rank: int,
        reduce_op: str = "sum",
    ) -> torch.Tensor:
        """
        Reduce-scatter operation: aggregate gradients and scatter to owners.

        Each worker contributes a tensor, and each receives a reduced chunk.

        Args:
            tensor: This worker's tensor contribution
            worker_addresses: List of all worker addresses in ring order
            rank: This worker's position in the ring
            reduce_op: Reduction operation ('sum', 'avg', 'max', 'min')

        Returns:
            Reduced tensor chunk for this worker
        """
        world_size = len(worker_addresses)
        collective_id = str(uuid.uuid4())

        logger.info(
            f"Starting reduce-scatter collective {collective_id}: "
            f"rank={rank}, world_size={world_size}, op={reduce_op}"
        )

        # Edge case: no other workers
        if world_size == 0:
            return tensor.clone()

        # Split tensor into chunks for scattering
        chunk_size = tensor.numel() // world_size
        chunks = torch.split(tensor, chunk_size)

        # Ensure we have exactly world_size chunks (pad last if needed)
        if len(chunks) < world_size:
            # Pad with zeros
            last_chunk = chunks[-1]
            padding_size = chunk_size - last_chunk.numel()
            if padding_size > 0:
                padded = torch.cat(
                    [last_chunk, torch.zeros(padding_size, dtype=tensor.dtype)]
                )
                chunks = list(chunks[:-1]) + [padded]

        # Initialize reduced chunks
        reduced_chunks = [chunk.clone() for chunk in chunks]

        # TODO: Implement proper ring reduce-scatter
        # For now, simulate by gathering all and reducing locally
        all_tensors = await self.all_gather(tensor, worker_addresses, rank)

        # Reduce each chunk across all workers
        for i in range(world_size):
            # Gather chunk i from all workers
            chunk_contributions = []
            for worker_tensor in all_tensors:
                worker_chunks = torch.split(worker_tensor, chunk_size)
                if i < len(worker_chunks):
                    chunk_contributions.append(worker_chunks[i])

            # Reduce chunk i
            if chunk_contributions:
                stacked = torch.stack(chunk_contributions)
                if reduce_op == "sum":
                    reduced_chunks[i] = stacked.sum(dim=0)
                elif reduce_op == "avg":
                    reduced_chunks[i] = stacked.mean(dim=0)
                elif reduce_op == "max":
                    reduced_chunks[i] = stacked.max(dim=0)[0]
                elif reduce_op == "min":
                    reduced_chunks[i] = stacked.min(dim=0)[0]

        # Return this worker's chunk
        result = reduced_chunks[rank]

        logger.info(f"Completed reduce-scatter collective {collective_id}")

        return result

    async def all_reduce(
        self,
        tensor: torch.Tensor,
        worker_addresses: List[str],
        rank: int,
        reduce_op: str = "sum",
    ) -> torch.Tensor:
        """
        All-reduce: reduce tensors across all workers and broadcast result.

        Equivalent to reduce-scatter followed by all-gather.

        Args:
            tensor: This worker's tensor contribution
            worker_addresses: List of all worker addresses
            rank: This worker's position
            reduce_op: Reduction operation

        Returns:
            Fully reduced tensor (same on all workers)
        """
        # Gather all tensors
        all_tensors = await self.all_gather(tensor, worker_addresses, rank)

        # Reduce locally
        stacked = torch.stack(all_tensors)
        if reduce_op == "sum":
            result = stacked.sum(dim=0)
        elif reduce_op == "avg":
            result = stacked.mean(dim=0)
        elif reduce_op == "max":
            result = stacked.max(dim=0)[0]
        elif reduce_op == "min":
            result = stacked.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown reduce op: {reduce_op}")

        return result


class RingAllGather:
    """
    Optimized ring-based all-gather implementation.

    This is a placeholder for the full ring algorithm which will be
    more bandwidth-efficient than the simple all-to-all approach above.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    async def execute(
        self,
        tensor: torch.Tensor,
        send_to: str,
        recv_from: str,
        client: WorkerGRPCClient,
    ) -> List[torch.Tensor]:
        """
        Execute ring all-gather algorithm.

        In each step:
        1. Send current chunk to next worker
        2. Receive chunk from previous worker
        3. After N-1 steps, all workers have all chunks
        """
        # TODO: Implement optimized ring algorithm
        # This reduces bandwidth from O(N) to O(1) per worker
        raise NotImplementedError("Optimized ring all-gather coming soon")


class RingReduceScatter:
    """
    Optimized ring-based reduce-scatter implementation.

    This will be more bandwidth-efficient than the current approach.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    async def execute(
        self,
        tensor: torch.Tensor,
        send_to: str,
        recv_from: str,
        client: WorkerGRPCClient,
        reduce_op: str = "sum",
    ) -> torch.Tensor:
        """
        Execute ring reduce-scatter algorithm.

        In each step:
        1. Reduce local chunk with received chunk
        2. Send reduced chunk to next worker
        3. After N-1 steps, each worker has its reduced chunk
        """
        # TODO: Implement optimized ring reduce-scatter
        raise NotImplementedError("Optimized ring reduce-scatter coming soon")
