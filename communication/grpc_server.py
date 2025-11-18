"""
gRPC server implementation for worker-to-worker communication.

Each worker runs a gRPC server to serve parameter requests and participate
in collective operations (all-gather, reduce-scatter).
"""

import asyncio
import logging
import time
from typing import Dict, Optional
import grpc
import torch

from communication.proto import worker_pb2, worker_pb2_grpc, tensor_pb2
from communication.serialization import (
    serialize_tensor,
    deserialize_tensor,
    chunk_tensor,
    serialize_tensor_compressed,
    deserialize_tensor_compressed,
)


logger = logging.getLogger(__name__)


class WorkerServicer(worker_pb2_grpc.WorkerServiceServicer):
    """
    gRPC servicer implementing the WorkerService protocol.

    This runs on each worker to handle parameter requests from peers.
    """

    def __init__(self, worker_id: str, parameter_store: Dict[str, torch.Tensor], enable_compression: bool = True):
        """
        Initialize the worker servicer.

        Args:
            worker_id: Unique identifier for this worker
            parameter_store: Dictionary mapping parameter names to tensors
                           (this worker's owned parameter shards)
            enable_compression: Whether to enable INT8 compression for transfers
        """
        self.worker_id = worker_id
        self.parameter_store = parameter_store
        self.enable_compression = enable_compression

        # Gradient accumulator: {version: {shard_id: {worker_id: gradients}}}
        # RESURRECTED for async parameter server architecture!
        # Now version-tracked for bounded staleness and adaptive aggregation
        self.gradient_accumulator: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        self.gradient_lock = asyncio.Lock()

        # Track expected number of gradient contributions per version
        # This will be set when training starts (world_size - 1, excluding self)
        self.expected_gradient_count = 0

        # Adaptive aggregation threshold (50-90% of workers)
        self.aggregation_threshold = 0.75  # Default: 75% of workers
        self.world_size = 1  # Will be set by training coordinator

        # Gradient arrival time tracking for adaptive timeout
        self.gradient_arrival_times: List[float] = []  # History of arrival times
        self.max_history_size = 100

        # Parameter version tracking
        self.parameter_versions: Dict[str, int] = {}  # param_name -> version

        # Ring message buffer: {operation_id: {step: chunk}}
        # Stores incoming ring chunks for retrieval by ring collective operations
        self.ring_buffer: Dict[str, Dict[int, torch.Tensor]] = {}
        self.ring_lock = asyncio.Lock()

        logger.info(f"Initialized WorkerServicer for worker {worker_id} (compression={'ON' if enable_compression else 'OFF'})")

    async def GetParameters(
        self, request: worker_pb2.ParameterRequest, context: grpc.aio.ServicerContext
    ) -> worker_pb2.ParameterResponse:
        """
        Handle parameter request from another worker.

        Returns the requested parameter shard if this worker owns it.
        """
        logger.debug(
            f"GetParameters request from {request.requester_id} "
            f"for shard [{request.shard_start}:{request.shard_end}]"
        )

        try:
            # Look up parameter by name or shard range
            # For now, we use parameter_name as the key
            if request.parameter_name not in self.parameter_store:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Parameter {request.parameter_name} not found")
                return worker_pb2.ParameterResponse()

            # Get the parameter tensor
            param_tensor = self.parameter_store[request.parameter_name]

            # Serialize to protobuf (with compression if enabled)
            tensor_proto = serialize_tensor_compressed(
                param_tensor,
                name=request.parameter_name,
                compress=self.enable_compression
            )

            # Create response
            response = worker_pb2.ParameterResponse()
            response.parameters.CopyFrom(tensor_proto)
            response.shard_start = request.shard_start
            response.shard_end = request.shard_end

            logger.debug(f"Sending parameters to {request.requester_id}")
            return response

        except Exception as e:
            logger.error(f"Error handling GetParameters: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return worker_pb2.ParameterResponse()

    async def StreamParameters(
        self, request: worker_pb2.ParameterRequest, context: grpc.aio.ServicerContext
    ):
        """
        Stream large parameters in chunks to avoid memory issues.

        Yields TensorChunk messages for efficient transfer.
        """
        logger.debug(
            f"StreamParameters request from {request.requester_id} "
            f"for {request.parameter_name}"
        )

        try:
            # Get parameter
            if request.parameter_name not in self.parameter_store:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Parameter {request.parameter_name} not found")
                return

            param_tensor = self.parameter_store[request.parameter_name]

            # Chunk the tensor (10MB chunks)
            chunks = chunk_tensor(param_tensor, chunk_size_mb=10, name=request.parameter_name)

            # Stream chunks
            for chunk in chunks:
                yield chunk
                # Small delay to avoid overwhelming the network
                await asyncio.sleep(0.001)

            logger.debug(
                f"Streamed {len(chunks)} chunks to {request.requester_id}"
            )

        except Exception as e:
            logger.error(f"Error streaming parameters: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def SendGradients(
        self, request: worker_pb2.GradientUpdate, context: grpc.aio.ServicerContext
    ) -> worker_pb2.GradientAck:
        """
        Receive gradient updates from another worker.

        Accumulates gradients for the specified training step and shard.
        When all workers have contributed, the accumulated gradients are ready
        for the optimizer step.
        """
        logger.debug(
            f"Received gradients from {request.sender_id} "
            f"for step {request.step}, shard [{request.shard_start}:{request.shard_end}]"
        )

        try:
            # Deserialize gradients (handles compression automatically)
            gradients = deserialize_tensor_compressed(request.gradients)

            # Accumulate gradients with proper synchronization
            async with self.gradient_lock:
                version = request.step  # Use step as version
                # Use parameter name from the tensor
                param_name = request.gradients.name if request.gradients.name else f"shard_{request.shard_start}_{request.shard_end}"

                # Initialize version accumulator if needed
                if version not in self.gradient_accumulator:
                    self.gradient_accumulator[version] = {}

                # Initialize parameter accumulator if needed
                if param_name not in self.gradient_accumulator[version]:
                    self.gradient_accumulator[version][param_name] = {}

                # Add this worker's gradient contribution (keyed by worker_id)
                self.gradient_accumulator[version][param_name][request.sender_id] = gradients

                num_contributions = len(self.gradient_accumulator[version][param_name])

                # Track arrival time for adaptive timeout
                if len(self.gradient_arrival_times) < self.max_history_size:
                    self.gradient_arrival_times.append(time.time())

                # Check if ready to aggregate
                threshold = max(1, int(self.world_size * self.aggregation_threshold))
                ready_to_aggregate = num_contributions >= threshold

                logger.debug(
                    f"Accumulated gradient {num_contributions}/{threshold} for version {version}, "
                    f"param {param_name}, shape {gradients.shape}"
                    f"{' [READY]' if ready_to_aggregate else ''}"
                )

            # Acknowledge
            ack = worker_pb2.GradientAck()
            ack.success = True
            ack.message = (
                f"Gradient received for version {version} "
                f"({num_contributions} total contributions)"
            )

            return ack

        except Exception as e:
            logger.error(f"Error receiving gradients: {e}")
            ack = worker_pb2.GradientAck()
            ack.success = False
            ack.message = str(e)
            return ack

    async def get_accumulated_gradients(
        self,
        version: int,
        param_name: str,
        reduce_op: str = "sum",
        wait_for_threshold: bool = False,
        timeout: float = 5.0
    ) -> Optional[torch.Tensor]:
        """
        Get accumulated gradients for a specific version and parameter.

        Aggregates all gradient contributions using the specified reduction operation.

        Args:
            version: Training step version
            param_name: Parameter name
            reduce_op: Reduction operation ('sum' or 'mean')
            wait_for_threshold: Whether to wait for aggregation threshold
            timeout: Maximum time to wait (seconds)

        Returns:
            Aggregated gradient tensor or None if not ready
        """
        start_time = time.time()

        while wait_for_threshold:
            async with self.gradient_lock:
                if version not in self.gradient_accumulator:
                    break

                if param_name not in self.gradient_accumulator[version]:
                    break

                contributions = self.gradient_accumulator[version][param_name]
                threshold = max(1, int(self.world_size * self.aggregation_threshold))

                if len(contributions) >= threshold:
                    # Ready to aggregate
                    break

            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(
                    f"Timeout waiting for gradients at version {version}, "
                    f"param {param_name} ({len(contributions)}/{threshold})"
                )
                break

            # Sleep briefly before checking again
            await asyncio.sleep(0.01)

        # Aggregate whatever we have
        async with self.gradient_lock:
            if version not in self.gradient_accumulator:
                return None

            if param_name not in self.gradient_accumulator[version]:
                return None

            contributions = self.gradient_accumulator[version][param_name]

            if len(contributions) == 0:
                return None

            # Aggregate gradients (now Dict[worker_id, tensor])
            gradients = list(contributions.values())

            if reduce_op == "sum":
                aggregated = torch.sum(torch.stack(gradients), dim=0)
            elif reduce_op == "mean":
                aggregated = torch.mean(torch.stack(gradients), dim=0)
            else:
                raise ValueError(f"Unknown reduce_op: {reduce_op}")

            logger.debug(
                f"Aggregated {len(gradients)} gradient contributions "
                f"for version {version}, param {param_name} using {reduce_op}"
            )

            return aggregated

    async def clear_gradients(self, step: int):
        """
        Clear accumulated gradients for a specific step.

        Called after optimizer step to free memory.

        Args:
            step: Training step number
        """
        async with self.gradient_lock:
            if step in self.gradient_accumulator:
                del self.gradient_accumulator[step]
                logger.debug(f"Cleared gradients for step {step}")

    def set_expected_gradient_count(self, count: int):
        """
        Set the expected number of gradient contributions per step.

        This should be world_size - 1 (all workers except self).

        Args:
            count: Expected number of contributions
        """
        self.expected_gradient_count = count
        logger.info(f"Set expected gradient count to {count}")

    def set_world_size(self, world_size: int):
        """
        Set the world size for adaptive aggregation.

        Args:
            world_size: Total number of workers
        """
        self.world_size = world_size
        logger.info(f"Set world size to {world_size}")

    def set_aggregation_threshold(self, threshold: float):
        """
        Set the aggregation threshold (fraction of workers).

        Args:
            threshold: Fraction of workers needed (0.5 = 50%, 0.75 = 75%, etc.)
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"Threshold must be in (0, 1], got {threshold}")

        self.aggregation_threshold = threshold
        logger.info(f"Set aggregation threshold to {threshold:.0%}")

    async def is_step_ready(self, step: int, shard_id: str) -> bool:
        """
        Check if all expected gradients have been received for a step.

        Args:
            step: Training step number
            shard_id: Shard identifier

        Returns:
            True if all expected gradients received, False otherwise
        """
        async with self.gradient_lock:
            if step not in self.gradient_accumulator:
                return False

            if shard_id not in self.gradient_accumulator[step]:
                return False

            num_contributions = len(self.gradient_accumulator[step][shard_id])
            return num_contributions >= self.expected_gradient_count

    async def Ping(
        self, request: worker_pb2.PingRequest, context: grpc.aio.ServicerContext
    ) -> worker_pb2.PingResponse:
        """
        Handle ping request for latency measurement.
        """
        import time

        response = worker_pb2.PingResponse()
        response.timestamp = request.timestamp
        response.server_time = int(time.time() * 1000)  # milliseconds

        return response

    async def SendRingChunk(
        self, request: worker_pb2.RingChunkRequest, context: grpc.aio.ServicerContext
    ) -> worker_pb2.RingChunkAck:
        """
        Receive a chunk in a ring collective operation.

        The chunk is stored in the ring buffer indexed by operation_id and step,
        where it can be retrieved by the ring collective algorithm.
        """
        logger.debug(
            f"Received ring chunk from {request.sender_id} "
            f"(op={request.operation_id}, step={request.step})"
        )

        try:
            # Deserialize chunk (handles compression automatically)
            chunk = deserialize_tensor_compressed(request.chunk)

            # Store in ring buffer with proper synchronization
            async with self.ring_lock:
                # Initialize operation buffer if needed
                if request.operation_id not in self.ring_buffer:
                    self.ring_buffer[request.operation_id] = {}

                # Store chunk for this step
                self.ring_buffer[request.operation_id][request.step] = chunk

                logger.debug(
                    f"Stored ring chunk for op={request.operation_id}, "
                    f"step={request.step}, shape={chunk.shape}"
                )

            # Acknowledge receipt
            ack = worker_pb2.RingChunkAck()
            ack.success = True
            ack.message = f"Ring chunk received for step {request.step}"
            return ack

        except Exception as e:
            logger.error(f"Error receiving ring chunk: {e}")
            ack = worker_pb2.RingChunkAck()
            ack.success = False
            ack.message = str(e)
            return ack

    async def get_ring_chunk(
        self, operation_id: str, step: int, timeout: float = 10.0
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a ring chunk from the buffer.

        This method blocks until the chunk arrives or timeout is reached.

        Args:
            operation_id: Unique ID for the ring operation
            step: Ring step number
            timeout: Maximum time to wait for chunk (seconds)

        Returns:
            The chunk tensor, or None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            async with self.ring_lock:
                if operation_id in self.ring_buffer:
                    if step in self.ring_buffer[operation_id]:
                        # Chunk available, return it
                        chunk = self.ring_buffer[operation_id][step]
                        logger.debug(
                            f"Retrieved ring chunk: op={operation_id}, "
                            f"step={step}, shape={chunk.shape}"
                        )
                        return chunk

            # Chunk not available yet, wait a bit
            await asyncio.sleep(0.01)  # 10ms polling interval

        # Timeout
        logger.warning(
            f"Timeout waiting for ring chunk: op={operation_id}, step={step}"
        )
        return None

    async def clear_ring_operation(self, operation_id: str):
        """
        Clear ring buffer for a completed operation.

        Args:
            operation_id: Unique ID for the ring operation to clear
        """
        async with self.ring_lock:
            if operation_id in self.ring_buffer:
                del self.ring_buffer[operation_id]
                logger.debug(f"Cleared ring buffer for operation {operation_id}")

    # All-gather and reduce-scatter will be implemented in collectives.py


class WorkerGRPCServer:
    """
    Manages the gRPC server lifecycle for a worker.
    """

    def __init__(
        self,
        worker_id: str,
        parameter_store: Dict[str, torch.Tensor],
        host: str = "0.0.0.0",
        port: int = 50051,
        enable_compression: bool = True,
    ):
        """
        Initialize gRPC server for this worker.

        Args:
            worker_id: Unique worker identifier
            parameter_store: Dictionary of owned parameters
            host: Host to bind to
            port: Port to listen on
            enable_compression: Whether to enable INT8 compression for transfers
        """
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.enable_compression = enable_compression
        self.server: Optional[grpc.aio.Server] = None
        self.servicer = WorkerServicer(worker_id, parameter_store, enable_compression=enable_compression)

    async def start(self):
        """Start the gRPC server."""
        # Configure server with larger message sizes for parameter transfer
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
        self.server = grpc.aio.server(options=options)
        worker_pb2_grpc.add_WorkerServiceServicer_to_server(self.servicer, self.server)

        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)

        await self.server.start()
        logger.info(f"Worker {self.worker_id} gRPC server started on {listen_addr}")

    async def stop(self, grace: Optional[float] = 5.0):
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info(f"Stopping gRPC server for worker {self.worker_id}")
            await self.server.stop(grace)
            logger.info(f"gRPC server stopped")

    async def wait_for_termination(self):
        """Wait for the server to terminate."""
        if self.server:
            await self.server.wait_for_termination()

    def update_parameters(self, parameter_name: str, tensor: torch.Tensor):
        """
        Update a parameter in the store.

        This is called after optimizer steps to keep served parameters fresh.
        """
        self.servicer.parameter_store[parameter_name] = tensor
