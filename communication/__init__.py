"""
Communication module for Legion distributed training.

Provides collective operations and transport layers for:
- All-gather: Collect parameters from all workers
- Reduce-scatter: Aggregate and distribute gradients
- Point-to-point: Direct worker-to-worker communication
"""

from communication.grpc_server import WorkerGRPCServer
from communication.grpc_client import WorkerGRPCClient
from communication.collectives import CollectiveOps
from communication.serialization import serialize_tensor, deserialize_tensor

__version__ = "0.1.0"

__all__ = [
    "WorkerGRPCServer",
    "WorkerGRPCClient",
    "CollectiveOps",
    "serialize_tensor",
    "deserialize_tensor",
]
