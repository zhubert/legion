"""
Communication module for Legion distributed training.

Provides collective operations and transport layers for:
- All-gather: Collect parameters from all workers
- Reduce-scatter: Aggregate and distribute gradients
- Point-to-point: Direct worker-to-worker communication
"""

__version__ = "0.1.0"
