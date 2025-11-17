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
from communication.serialization import serialize_tensor, deserialize_tensor, chunk_tensor


logger = logging.getLogger(__name__)


class WorkerServicer(worker_pb2_grpc.WorkerServiceServicer):
    """
    gRPC servicer implementing the WorkerService protocol.

    This runs on each worker to handle parameter requests from peers.
    """

    def __init__(self, worker_id: str, parameter_store: Dict[str, torch.Tensor]):
        """
        Initialize the worker servicer.

        Args:
            worker_id: Unique identifier for this worker
            parameter_store: Dictionary mapping parameter names to tensors
                           (this worker's owned parameter shards)
        """
        self.worker_id = worker_id
        self.parameter_store = parameter_store

        # Gradient accumulator: {step: {shard_id: [gradients from each worker]}}
        self.gradient_accumulator: Dict[int, Dict[str, list]] = {}
        self.gradient_lock = asyncio.Lock()

        # Track expected number of gradient contributions per step
        # This will be set when training starts (world_size - 1, excluding self)
        self.expected_gradient_count = 0

        logger.info(f"Initialized WorkerServicer for worker {worker_id}")

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

            # Serialize to protobuf
            tensor_proto = serialize_tensor(param_tensor, name=request.parameter_name)

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
            # Deserialize gradients
            gradients = deserialize_tensor(request.gradients)

            # Accumulate gradients with proper synchronization
            async with self.gradient_lock:
                step = request.step
                shard_id = f"{request.shard_start}_{request.shard_end}"

                # Initialize step accumulator if needed
                if step not in self.gradient_accumulator:
                    self.gradient_accumulator[step] = {}

                # Initialize shard accumulator if needed
                if shard_id not in self.gradient_accumulator[step]:
                    self.gradient_accumulator[step][shard_id] = []

                # Add this worker's gradient contribution
                self.gradient_accumulator[step][shard_id].append({
                    'sender_id': request.sender_id,
                    'gradients': gradients,
                    'timestamp': time.time()
                })

                num_contributions = len(self.gradient_accumulator[step][shard_id])

                logger.debug(
                    f"Accumulated gradient {num_contributions} for step {step}, "
                    f"shard {shard_id}"
                )

            # Acknowledge
            ack = worker_pb2.GradientAck()
            ack.success = True
            ack.message = (
                f"Gradient received for step {step} "
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
        step: int,
        shard_id: str,
        reduce_op: str = "sum"
    ) -> Optional[torch.Tensor]:
        """
        Get accumulated gradients for a specific step and shard.

        Aggregates all gradient contributions using the specified reduction operation.

        Args:
            step: Training step number
            shard_id: Shard identifier (format: "start_end")
            reduce_op: Reduction operation ('sum' or 'mean')

        Returns:
            Aggregated gradient tensor or None if not ready
        """
        async with self.gradient_lock:
            if step not in self.gradient_accumulator:
                return None

            if shard_id not in self.gradient_accumulator[step]:
                return None

            contributions = self.gradient_accumulator[step][shard_id]

            if len(contributions) == 0:
                return None

            # Aggregate gradients
            gradients = [c['gradients'] for c in contributions]

            if reduce_op == "sum":
                aggregated = torch.sum(torch.stack(gradients), dim=0)
            elif reduce_op == "mean":
                aggregated = torch.mean(torch.stack(gradients), dim=0)
            else:
                raise ValueError(f"Unknown reduce_op: {reduce_op}")

            logger.debug(
                f"Aggregated {len(gradients)} gradient contributions "
                f"for step {step}, shard {shard_id} using {reduce_op}"
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
    ):
        """
        Initialize gRPC server for this worker.

        Args:
            worker_id: Unique worker identifier
            parameter_store: Dictionary of owned parameters
            host: Host to bind to
            port: Port to listen on
        """
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.server: Optional[grpc.aio.Server] = None
        self.servicer = WorkerServicer(worker_id, parameter_store)

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
