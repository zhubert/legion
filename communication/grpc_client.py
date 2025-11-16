"""
gRPC client for worker-to-worker communication.

Workers use this client to request parameters from peers and send gradients.
"""

import asyncio
import logging
from typing import Optional, List
import grpc
import torch
import time

from communication.proto import worker_pb2, worker_pb2_grpc
from communication.serialization import (
    serialize_tensor,
    deserialize_tensor,
    reassemble_chunks,
)


logger = logging.getLogger(__name__)


class WorkerGRPCClient:
    """
    Client for making gRPC requests to other workers.
    """

    def __init__(self, worker_id: str, timeout: float = 30.0):
        """
        Initialize the gRPC client.

        Args:
            worker_id: This worker's unique identifier
            timeout: Default timeout for RPC calls in seconds
        """
        self.worker_id = worker_id
        self.timeout = timeout
        self.channels = {}  # Cache of open channels to other workers
        logger.info(f"Initialized WorkerGRPCClient for worker {worker_id}")

    def get_channel(self, worker_address: str) -> grpc.aio.Channel:
        """
        Get or create a gRPC channel to another worker.

        Args:
            worker_address: Address in format "host:port"

        Returns:
            gRPC async channel
        """
        if worker_address not in self.channels:
            # Configure channel with larger message sizes for parameter transfer
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ]
            self.channels[worker_address] = grpc.aio.insecure_channel(
                worker_address,
                options=options
            )
            logger.debug(f"Created channel to {worker_address}")

        return self.channels[worker_address]

    async def get_parameters(
        self,
        worker_address: str,
        parameter_name: str,
        shard_start: int = 0,
        shard_end: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Request parameters from another worker.

        Args:
            worker_address: Target worker address "host:port"
            parameter_name: Name of the parameter to fetch
            shard_start: Start index of shard
            shard_end: End index of shard

        Returns:
            Parameter tensor or None if request failed
        """
        try:
            channel = self.get_channel(worker_address)
            stub = worker_pb2_grpc.WorkerServiceStub(channel)

            # Create request
            request = worker_pb2.ParameterRequest()
            request.requester_id = self.worker_id
            request.parameter_name = parameter_name
            request.shard_start = shard_start
            request.shard_end = shard_end

            # Make RPC call
            response = await stub.GetParameters(request, timeout=self.timeout)

            # Deserialize tensor
            tensor = deserialize_tensor(response.parameters)

            logger.debug(
                f"Fetched {parameter_name} from {worker_address}, "
                f"shape {tensor.shape}"
            )

            return tensor

        except grpc.RpcError as e:
            logger.error(
                f"RPC error fetching parameters from {worker_address}: "
                f"{e.code()} - {e.details()}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching parameters from {worker_address}: {e}")
            return None

    async def stream_parameters(
        self,
        worker_address: str,
        parameter_name: str,
        shard_start: int = 0,
        shard_end: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Stream large parameters from another worker in chunks.

        Args:
            worker_address: Target worker address "host:port"
            parameter_name: Name of the parameter to fetch
            shard_start: Start index of shard
            shard_end: End index of shard

        Returns:
            Reassembled parameter tensor or None if failed
        """
        try:
            channel = self.get_channel(worker_address)
            stub = worker_pb2_grpc.WorkerServiceStub(channel)

            # Create request
            request = worker_pb2.ParameterRequest()
            request.requester_id = self.worker_id
            request.parameter_name = parameter_name
            request.shard_start = shard_start
            request.shard_end = shard_end

            # Stream chunks
            chunks = []
            async for chunk in stub.StreamParameters(request, timeout=self.timeout):
                chunks.append(chunk)

            # Reassemble tensor
            tensor = reassemble_chunks(chunks)

            logger.debug(
                f"Streamed {parameter_name} from {worker_address}, "
                f"shape {tensor.shape} ({len(chunks)} chunks)"
            )

            return tensor

        except grpc.RpcError as e:
            logger.error(
                f"RPC error streaming parameters from {worker_address}: "
                f"{e.code()} - {e.details()}"
            )
            return None
        except Exception as e:
            logger.error(f"Error streaming parameters from {worker_address}: {e}")
            return None

    async def send_gradients(
        self,
        worker_address: str,
        gradients: torch.Tensor,
        step: int,
        shard_start: int = 0,
        shard_end: int = -1,
    ) -> bool:
        """
        Send gradient updates to a parameter owner.

        Args:
            worker_address: Target worker address "host:port"
            gradients: Gradient tensor to send
            step: Training step number
            shard_start: Start index of shard
            shard_end: End index of shard

        Returns:
            True if acknowledged, False otherwise
        """
        try:
            channel = self.get_channel(worker_address)
            stub = worker_pb2_grpc.WorkerServiceStub(channel)

            # Serialize gradients
            grad_proto = serialize_tensor(gradients)

            # Create request
            request = worker_pb2.GradientUpdate()
            request.sender_id = self.worker_id
            request.gradients.CopyFrom(grad_proto)
            request.step = step
            request.shard_start = shard_start
            request.shard_end = shard_end

            # Send
            response = await stub.SendGradients(request, timeout=self.timeout)

            if response.success:
                logger.debug(f"Sent gradients to {worker_address} for step {step}")
                return True
            else:
                logger.warning(
                    f"Failed to send gradients to {worker_address}: "
                    f"{response.message}"
                )
                return False

        except grpc.RpcError as e:
            logger.error(
                f"RPC error sending gradients to {worker_address}: "
                f"{e.code()} - {e.details()}"
            )
            return False
        except Exception as e:
            logger.error(f"Error sending gradients to {worker_address}: {e}")
            return False

    async def ping(self, worker_address: str) -> Optional[float]:
        """
        Measure latency to another worker.

        Args:
            worker_address: Target worker address "host:port"

        Returns:
            Round-trip time in milliseconds, or None if failed
        """
        try:
            channel = self.get_channel(worker_address)
            stub = worker_pb2_grpc.WorkerServiceStub(channel)

            # Create ping request
            request = worker_pb2.PingRequest()
            request.worker_id = self.worker_id
            request.timestamp = int(time.time() * 1000)

            # Measure RTT
            start = time.time()
            response = await stub.Ping(request, timeout=5.0)
            end = time.time()

            rtt_ms = (end - start) * 1000

            logger.debug(f"Ping to {worker_address}: {rtt_ms:.2f}ms")

            return rtt_ms

        except grpc.RpcError as e:
            logger.error(
                f"RPC error pinging {worker_address}: {e.code()} - {e.details()}"
            )
            return None
        except Exception as e:
            logger.error(f"Error pinging {worker_address}: {e}")
            return None

    async def close(self):
        """Close all open channels."""
        for address, channel in self.channels.items():
            await channel.close()
            logger.debug(f"Closed channel to {address}")

        self.channels.clear()
        logger.info(f"WorkerGRPCClient closed all channels")
