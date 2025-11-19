"""
Parameter shard manager for worker nodes.

Manages the worker's assigned partition of model parameters, including
loading, updating, and checkpointing.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn

from core.partitioner import Partitioner, ParameterPartition


logger = logging.getLogger(__name__)


class ShardManager:
    """
    Manages parameter shard for a worker.

    Handles loading, storing, updating, and checkpointing the worker's
    assigned portion of the model parameters.
    """

    def __init__(
        self,
        worker_id: str,
        model: nn.Module,
        shard_start: int,
        shard_end: int,
        device: str = "cpu",
        checkpoint_dir: str = "./checkpoints",
        rank: int = 0,
        world_size: int = 1,
        partition: Optional[ParameterPartition] = None
    ):
        """
        Initialize shard manager.

        Args:
            worker_id: Unique worker identifier
            model: PyTorch model
            shard_start: Start index of parameter shard
            shard_end: End index of parameter shard
            device: Device to store parameters ('cpu' or 'cuda')
            checkpoint_dir: Directory for checkpoints
            rank: Worker rank in the training group
            world_size: Total number of workers
            partition: Optional pre-computed ParameterPartition
        """
        self.worker_id = worker_id
        self.model = model
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.world_size = world_size

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use provided partition or create default
        if partition is not None:
            self.partition = partition
        else:
            self.partition = ParameterPartition(
                rank=rank,
                world_size=world_size,
                param_names=[],  # Will be populated during load
                start_idx=shard_start,
                end_idx=shard_end
            )

        # Shard state
        self._shard_loaded = False
        self._owned_parameters: Optional[torch.Tensor] = None
        self._optimizer_state: Optional[Dict] = None

        logger.info(
            f"Shard manager initialized: worker={worker_id}, "
            f"shard=[{shard_start}:{shard_end}] ({shard_end - shard_start} params)"
        )

    def load_shard(self, checkpoint_path: Optional[str] = None):
        """
        Load parameter shard.

        If checkpoint provided, loads from checkpoint. Otherwise initializes
        from model.

        Args:
            checkpoint_path: Optional path to checkpoint file
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)
        else:
            self._initialize_from_model()

        self._shard_loaded = True
        logger.info(f"Shard loaded successfully")

    def _initialize_from_model(self):
        """Initialize shard from model parameters."""
        # Flatten all model parameters
        all_params = []
        param_names = []

        for name, param in self.model.named_parameters():
            all_params.append(param.data.flatten())
            param_names.append(name)

        # Concatenate into single tensor
        flat_params = torch.cat(all_params)

        # Extract our shard
        self._owned_parameters = flat_params[self.shard_start:self.shard_end].clone()
        self._owned_parameters = self._owned_parameters.to(self.device)

        # Store parameter names for this shard
        self.partition.param_names = param_names

        logger.debug(
            f"Initialized shard from model: {self._owned_parameters.shape}, "
            f"device={self._owned_parameters.device}"
        )

    def _load_from_checkpoint(self, checkpoint_path: str):
        """
        Load shard from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading shard from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self._owned_parameters = checkpoint['parameters']
        self._optimizer_state = checkpoint.get('optimizer_state')
        self.partition.param_names = checkpoint.get('param_names', [])

        logger.info(
            f"Loaded checkpoint: {self._owned_parameters.shape}, "
            f"step={checkpoint.get('global_step', 'unknown')}"
        )

    def get_owned_parameters(self) -> torch.Tensor:
        """
        Get owned parameters tensor.

        Returns:
            Tensor of owned parameters

        Raises:
            RuntimeError: If shard not loaded
        """
        if not self._shard_loaded or self._owned_parameters is None:
            raise RuntimeError("Shard not loaded. Call load_shard() first.")

        return self._owned_parameters

    def update_parameters(self, new_params: torch.Tensor):
        """
        Update owned parameters.

        Args:
            new_params: New parameter values

        Raises:
            RuntimeError: If shard not loaded
            ValueError: If shape mismatch
        """
        if not self._shard_loaded or self._owned_parameters is None:
            raise RuntimeError("Shard not loaded. Call load_shard() first.")

        if new_params.shape != self._owned_parameters.shape:
            raise ValueError(
                f"Shape mismatch: expected {self._owned_parameters.shape}, "
                f"got {new_params.shape}"
            )

        self._owned_parameters.copy_(new_params)
        logger.debug("Parameters updated")

    def save_checkpoint(
        self,
        global_step: int,
        optimizer_state: Optional[Dict] = None,
        keep_last_n: int = 3
    ) -> str:
        """
        Save checkpoint (legacy method - saves to worker-specific file).

        Args:
            global_step: Current training step
            optimizer_state: Optional optimizer state
            keep_last_n: Keep only last N checkpoints

        Returns:
            Path to saved checkpoint

        Raises:
            RuntimeError: If shard not loaded
        """
        if not self._shard_loaded or self._owned_parameters is None:
            raise RuntimeError("Shard not loaded. Call load_shard() first.")

        # Create checkpoint filename
        checkpoint_name = f"worker_{self.worker_id}_step_{global_step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        checkpoint = {
            'worker_id': self.worker_id,
            'global_step': global_step,
            'shard_start': self.shard_start,
            'shard_end': self.shard_end,
            'parameters': self._owned_parameters.cpu(),
            'optimizer_state': optimizer_state,
            'param_names': self.partition.param_names
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints(keep_last_n)

        return str(checkpoint_path)

    def save_shard_to_checkpoint(
        self,
        checkpoint_dir: str,
        global_step: int,
        rank: int,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_state: Optional[Dict] = None
    ) -> str:
        """
        Save shard to a shared checkpoint directory for assembly.

        This method saves the worker's parameter shard with full metadata needed
        for checkpoint assembly. Unlike save_checkpoint(), this creates a
        standardized format that the assembler can use to reconstruct the full model.

        Args:
            checkpoint_dir: Base checkpoint directory (e.g., './checkpoints')
            global_step: Current training step
            rank: Worker's rank in the training group
            model_config: Optional model configuration metadata
            optimizer_state: Optional optimizer state

        Returns:
            Path to saved shard file

        Raises:
            RuntimeError: If shard not loaded
        """
        if not self._shard_loaded or self._owned_parameters is None:
            raise RuntimeError("Shard not loaded. Call load_shard() first.")

        # Create step-specific checkpoint directory
        step_dir = Path(checkpoint_dir) / f"step_{global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Create shard filename
        shard_name = f"shard_rank_{rank}.pt"
        shard_path = step_dir / shard_name

        # Build shard data with full partition metadata
        shard_data = {
            # Identification
            'worker_id': self.worker_id,
            'rank': rank,
            'global_step': global_step,

            # Parameter data
            'parameters': self._owned_parameters.cpu(),
            'optimizer_state': optimizer_state,

            # Partition metadata (critical for assembly)
            'partition': {
                'shard_start': self.shard_start,
                'shard_end': self.shard_end,
                'num_params': self.shard_end - self.shard_start,
                'param_names': self.partition.param_names,
                'param_slices': self.partition.param_slices,
                'world_size': self.partition.world_size,
            },

            # Model configuration (for validation)
            'model_config': model_config,
        }

        # Save shard
        torch.save(shard_data, shard_path)
        logger.info(
            f"Shard saved to checkpoint: {shard_path} "
            f"(rank={rank}, params={self.shard_end - self.shard_start})"
        )

        return str(shard_path)

    def _cleanup_old_checkpoints(self, keep_last_n: int):
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of checkpoints to keep
        """
        # Find all checkpoints for this worker
        pattern = f"worker_{self.worker_id}_step_*.pt"
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )

        # Remove oldest checkpoints
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage in GB
        """
        if not self._shard_loaded or self._owned_parameters is None:
            return {
                'parameters_gb': 0.0,
                'total_gb': 0.0
            }

        # Calculate parameter memory
        param_bytes = self._owned_parameters.nelement() * self._owned_parameters.element_size()
        param_gb = param_bytes / (1024 ** 3)

        usage = {
            'parameters_gb': param_gb,
            'total_gb': param_gb
        }

        # Add optimizer state memory if available
        if self._optimizer_state:
            # Estimate optimizer memory (rough approximation)
            # Typically 2x parameters for Adam (momentum + variance)
            usage['optimizer_gb'] = param_gb * 2
            usage['total_gb'] += usage['optimizer_gb']

        # Add GPU memory if on CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            usage['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            usage['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)

        return usage

    def get_shard_info(self) -> Dict[str, Any]:
        """
        Get shard information.

        Returns:
            Dictionary with shard details
        """
        return {
            'worker_id': self.worker_id,
            'shard_start': self.shard_start,
            'shard_end': self.shard_end,
            'num_params': self.shard_end - self.shard_start,
            'loaded': self._shard_loaded,
            'device': self.device,
            'memory_usage': self.get_memory_usage()
        }

    def is_loaded(self) -> bool:
        """
        Check if shard is loaded.

        Returns:
            True if shard loaded, False otherwise
        """
        return self._shard_loaded

    def update_partition_info(self, partition: ParameterPartition):
        """
        Update partition information after initialization.

        This is called by the trainer after the Partitioner creates the actual
        parameter partitions. The ShardManager is initially created with placeholder
        ranges (0, total_params), then updated with the real partition info.

        Args:
            partition: The actual ParameterPartition for this worker
        """
        self.partition = partition
        self.shard_start = partition.start_idx
        self.shard_end = partition.end_idx
        logger.info(
            f"Updated partition info: shard=[{self.shard_start}:{self.shard_end}] "
            f"({self.shard_end - self.shard_start} params)"
        )
