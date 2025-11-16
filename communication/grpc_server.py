"""
gRPC server implementation for worker-to-worker communication.

Each worker runs a gRPC server to serve parameter requests and participate
in collective operations (all-gather, reduce-scatter).
"""

import asyncio
import logging
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

        In a real implementation, this would accumulate gradients for parameter updates.
        For now, we just acknowledge receipt.
        """
        logger.debug(
            f"Received gradients from {request.sender_id} "
            f"for step {request.step}"
        )

        try:
            # Deserialize gradients
            gradients = deserialize_tensor(request.gradients)

            # TODO: Accumulate gradients for optimizer step
            # This will be implemented in the training loop integration

            # Acknowledge
            ack = worker_pb2.GradientAck()
            ack.success = True
            ack.message = f"Gradients received for step {request.step}"

            return ack

        except Exception as e:
            logger.error(f"Error receiving gradients: {e}")
            ack = worker_pb2.GradientAck()
            ack.success = False
            ack.message = str(e)
            return ack

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
