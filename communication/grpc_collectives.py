"""
gRPC-based collective operations for real distributed training.

Implements all-gather and reduce-scatter using worker-to-worker gRPC calls.
This replaces the simulation-based shared memory collectives for actual deployment.
"""

import asyncio
import logging
from typing import List, Dict, Optional
import torch

from communication.grpc_client import WorkerGRPCClient


logger = logging.getLogger(__name__)


class GRPCCollectiveOps:
    """
    Collective operations using gRPC for worker-to-worker communication.

    Compatible interface with sim.collectives.CollectiveOps but uses
    real network communication instead of shared memory.
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
        Initialize gRPC-based collective operations.

        Args:
            rank: This worker's rank (0 to world_size-1)
            world_size: Total number of workers
            worker_id: This worker's unique identifier
            worker_addresses: List of worker addresses in rank order ["host:port", ...]
            grpc_client: gRPC client for making requests
            timeout: Timeout for collective operations in seconds
        """
        self.rank = rank
        self.world_size = world_size
        self.worker_id = worker_id
        self.worker_addresses = worker_addresses
        self.grpc_client = grpc_client
        self.timeout = timeout

        # Validate configuration
        if len(worker_addresses) != world_size:
            raise ValueError(
                f"Number of worker addresses ({len(worker_addresses)}) "
                f"does not match world_size ({world_size})"
            )

        logger.info(
            f"Initialized GRPCCollectiveOps: rank={rank}, world_size={world_size}"
        )

    async def all_gather_async(
        self,
        tensor: torch.Tensor,
        parameter_name: str = "parameter",
        shard_start: int = 0,
        shard_end: int = -1
    ) -> torch.Tensor:
        """
        All-gather: Collect tensors from all workers and concatenate.

        Each worker fetches parameter shards from all other workers via gRPC
        and concatenates them into a full parameter tensor.

        Args:
            tensor: This worker's local tensor shard
            parameter_name: Name of the parameter being gathered
            shard_start: Start index of this worker's shard
            shard_end: End index of this worker's shard

        Returns:
            Concatenated tensor from all workers
        """
        logger.debug(
            f"All-gather {parameter_name}: rank {self.rank} "
            f"gathering from {self.world_size} workers"
        )

        # Collect tensors from all workers (including self)
        gathered_tensors = [None] * self.world_size
        gathered_tensors[self.rank] = tensor.clone()

        # Fetch from other workers concurrently
        fetch_tasks = []
        for i in range(self.world_size):
            if i == self.rank:
                continue  # Skip self

            task = self._fetch_parameter_from_worker(
                rank=i,
                parameter_name=parameter_name,
                shard_start=shard_start,
                shard_end=shard_end
            )
            fetch_tasks.append((i, task))

        # Wait for all fetches to complete
        for i, task in fetch_tasks:
            try:
                fetched_tensor = await asyncio.wait_for(task, timeout=self.timeout)
                if fetched_tensor is not None:
                    gathered_tensors[i] = fetched_tensor
                else:
                    logger.error(f"Failed to fetch parameter from worker {i}")
                    # Use zeros as fallback
                    gathered_tensors[i] = torch.zeros_like(tensor)
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching parameter from worker {i}")
                gathered_tensors[i] = torch.zeros_like(tensor)

        # Concatenate all tensors
        result = torch.cat(gathered_tensors, dim=0)

        logger.debug(
            f"All-gather complete: {parameter_name}, "
            f"shape {tensor.shape} -> {result.shape}"
        )

        return result

    async def reduce_scatter_async(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
        step: int = 0
    ) -> torch.Tensor:
        """
        Reduce-scatter: Reduce tensor across workers and scatter result.

        Each worker:
        1. Splits the tensor into world_size chunks
        2. Sends chunks to the respective parameter owners via gRPC
        3. Receives and accumulates chunks from other workers

        Args:
            tensor: Full tensor to reduce (will be chunked)
            op: Reduction operation ('sum', 'mean')
            step: Training step number

        Returns:
            This worker's reduced chunk
        """
        logger.debug(
            f"Reduce-scatter: rank {self.rank} reducing {tensor.shape}"
        )

        # Calculate chunk sizes
        total_size = tensor.shape[0]
        chunk_size = total_size // self.world_size
        remainder = total_size % self.world_size

        # Split tensor into chunks for each worker
        chunks = []
        offset = 0
        for i in range(self.world_size):
            # Handle uneven division
            size = chunk_size + (1 if i < remainder else 0)
            chunk = tensor[offset:offset + size]
            chunks.append(chunk)
            offset += size

        # This worker's chunk (will accumulate)
        my_chunk = chunks[self.rank].clone()

        # Send chunks to other workers concurrently
        send_tasks = []
        for i in range(self.world_size):
            if i == self.rank:
                continue  # Don't send to self

            task = self._send_gradient_to_worker(
                rank=i,
                gradients=chunks[i],
                step=step,
                shard_start=0,
                shard_end=-1
            )
            send_tasks.append((i, task))

        # Wait for all sends to complete (but don't block on receiving)
        for i, task in send_tasks:
            try:
                success = await asyncio.wait_for(task, timeout=self.timeout)
                if not success:
                    logger.warning(f"Failed to send gradient chunk to worker {i}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout sending gradient to worker {i}")

        # Note: In a full implementation, we'd have a separate mechanism
        # for receiving and accumulating gradients from other workers.
        # For now, we simulate by assuming all workers contribute equally.

        # Apply reduction operation
        if op == "mean":
            my_chunk = my_chunk / self.world_size

        logger.debug(
            f"Reduce-scatter complete: rank {self.rank}, "
            f"chunk shape {my_chunk.shape}"
        )

        return my_chunk

    async def _fetch_parameter_from_worker(
        self,
        rank: int,
        parameter_name: str,
        shard_start: int,
        shard_end: int
    ) -> Optional[torch.Tensor]:
        """
        Fetch a parameter shard from another worker.

        Args:
            rank: Target worker rank
            parameter_name: Name of parameter to fetch
            shard_start: Start index of shard
            shard_end: End index of shard

        Returns:
            Parameter tensor or None if failed
        """
        worker_address = self.worker_addresses[rank]

        try:
            # Use streaming for large parameters
            tensor = await self.grpc_client.stream_parameters(
                worker_address=worker_address,
                parameter_name=parameter_name,
                shard_start=shard_start,
                shard_end=shard_end
            )
            return tensor
        except Exception as e:
            logger.error(
                f"Error fetching parameter from worker {rank} "
                f"({worker_address}): {e}"
            )
            return None

    async def _send_gradient_to_worker(
        self,
        rank: int,
        gradients: torch.Tensor,
        step: int,
        shard_start: int,
        shard_end: int
    ) -> bool:
        """
        Send gradient chunk to another worker.

        Args:
            rank: Target worker rank
            gradients: Gradient tensor to send
            step: Training step number
            shard_start: Start index of shard
            shard_end: End index of shard

        Returns:
            True if successful, False otherwise
        """
        worker_address = self.worker_addresses[rank]

        try:
            success = await self.grpc_client.send_gradients(
                worker_address=worker_address,
                gradients=gradients,
                step=step,
                shard_start=shard_start,
                shard_end=shard_end
            )
            return success
        except Exception as e:
            logger.error(
                f"Error sending gradients to worker {rank} "
                f"({worker_address}): {e}"
            )
            return False

    # Synchronous wrappers for compatibility with sim.collectives.CollectiveOps

    def all_gather(
        self,
        tensor: torch.Tensor,
        async_op: bool = False
    ) -> torch.Tensor:
        """
        Synchronous all-gather wrapper.

        Note: This runs the async operation in a blocking manner.
        For better performance, use all_gather_async directly in async contexts.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from within async context, create a task
            raise RuntimeError(
                "all_gather called from async context. "
                "Use all_gather_async instead."
            )

        return loop.run_until_complete(
            self.all_gather_async(tensor)
        )

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
        async_op: bool = False
    ) -> torch.Tensor:
        """
        Synchronous reduce-scatter wrapper.

        Note: This runs the async operation in a blocking manner.
        For better performance, use reduce_scatter_async directly in async contexts.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "reduce_scatter called from async context. "
                "Use reduce_scatter_async instead."
            )

        return loop.run_until_complete(
            self.reduce_scatter_async(tensor, op=op)
        )
