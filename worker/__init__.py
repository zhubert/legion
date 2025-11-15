"""
Worker module for Legion distributed training.

Workers are the compute nodes that:
- Store and update assigned parameter shards
- Execute forward/backward passes
- Participate in collective communications
- Report telemetry to coordinator
"""

__version__ = "0.1.0"
