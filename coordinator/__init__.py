"""
Coordinator module for Legion distributed training.

The coordinator is responsible for:
- Worker registration and discovery
- Health monitoring via heartbeat protocol
- Regional cluster assignment based on latency
- Checkpoint coordination
- Telemetry aggregation
"""

__version__ = "0.1.0"
