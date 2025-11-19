"""
Training configuration management for Legion coordinator.

The coordinator owns all training decisions including:
- Dataset selection and sharding
- Model architecture
- Hyperparameters (batch size, learning rate, etc.)
- Compression settings
- Training duration

Workers fetch their specific configuration from the coordinator on startup.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Global training configuration managed by coordinator.

    All workers must follow this configuration for consistent distributed training.
    """

    # Model configuration
    model_size: str = "tiny"  # "tiny", "small", "medium"

    # Dataset configuration
    dataset_type: str = "huggingface"  # "dummy", "distributed_dummy", "huggingface"
    dataset_name: Optional[str] = "tiny_shakespeare"  # e.g., "fineweb-edu", "pile", "shakespeare"
    tokenizer_name: Optional[str] = None  # Override default tokenizer

    # Training hyperparameters
    batch_size: int = 4  # Per-worker batch size (can be customized per worker)
    seq_len: int = 128  # Sequence length (must be consistent across workers)
    learning_rate: float = 0.001
    num_steps: int = 100  # Total training steps

    # Compression
    compression: str = "none"  # "none", "int8", "topk"

    # Checkpointing
    checkpoint_interval_steps: int = 100

    # Sharding strategy
    sharding_strategy: str = "interleaved"  # "interleaved", "contiguous"

    # Optional: Worker-specific overrides
    # Key: worker_id, Value: dict of overrides (e.g., {"batch_size": 8})
    worker_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TrainingConfig':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, path: str) -> 'TrainingConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def to_json_file(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_worker_config(self, worker_id: str, rank: int, world_size: int) -> Dict[str, Any]:
        """
        Get configuration for a specific worker.

        Applies worker-specific overrides if present.

        Args:
            worker_id: Worker identifier
            rank: Worker rank in training cluster
            world_size: Total number of workers

        Returns:
            Configuration dict for this worker
        """
        config = self.to_dict()

        # Add rank and world_size for dataset sharding
        config['rank'] = rank
        config['world_size'] = world_size

        # Apply worker-specific overrides
        if worker_id in self.worker_overrides:
            overrides = self.worker_overrides[worker_id]
            logger.info(f"Applying overrides for worker {worker_id}: {overrides}")
            config.update(overrides)

        return config

    def set_worker_override(self, worker_id: str, key: str, value: Any):
        """
        Set worker-specific configuration override.

        Args:
            worker_id: Worker identifier
            key: Configuration key to override
            value: Override value
        """
        if worker_id not in self.worker_overrides:
            self.worker_overrides[worker_id] = {}

        self.worker_overrides[worker_id][key] = value
        logger.info(f"Set override for {worker_id}: {key} = {value}")

    def validate(self) -> List[str]:
        """
        Validate training configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Model size validation
        if self.model_size not in ["tiny", "small", "medium"]:
            errors.append(f"Invalid model_size: {self.model_size}")

        # Dataset type validation
        if self.dataset_type not in ["dummy", "distributed_dummy", "huggingface"]:
            errors.append(f"Invalid dataset_type: {self.dataset_type}")

        # HuggingFace dataset requires dataset_name
        if self.dataset_type == "huggingface" and not self.dataset_name:
            errors.append("dataset_name required when dataset_type is 'huggingface'")

        # Hyperparameter validation
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive: {self.batch_size}")

        if self.seq_len <= 0:
            errors.append(f"seq_len must be positive: {self.seq_len}")

        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive: {self.learning_rate}")

        if self.num_steps <= 0:
            errors.append(f"num_steps must be positive: {self.num_steps}")

        # Compression validation
        if self.compression not in ["none", "int8", "topk"]:
            errors.append(f"Invalid compression: {self.compression}")

        return errors


class TrainingConfigManager:
    """
    Manages training configuration for the coordinator.

    Handles configuration storage, updates, and worker assignment.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize training config manager.

        Args:
            config: Initial training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        logger.info(f"Training config initialized: {self.config.model_size} model, "
                   f"{self.config.dataset_type} dataset, "
                   f"{self.config.num_steps} steps")

    def get_config(self) -> TrainingConfig:
        """Get current training configuration."""
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> List[str]:
        """
        Update training configuration.

        Args:
            updates: Dictionary of fields to update

        Returns:
            List of validation errors (empty if successful)
        """
        # Create new config with updates
        config_dict = self.config.to_dict()
        config_dict.update(updates)

        new_config = TrainingConfig.from_dict(config_dict)

        # Validate
        errors = new_config.validate()
        if errors:
            logger.error(f"Config validation failed: {errors}")
            return errors

        # Apply if valid
        self.config = new_config
        logger.info(f"Training config updated: {updates}")
        return []

    def get_worker_assignment(
        self,
        worker_id: str,
        rank: int,
        world_size: int
    ) -> Dict[str, Any]:
        """
        Get training configuration assignment for a specific worker.

        Args:
            worker_id: Worker identifier
            rank: Worker rank in cluster
            world_size: Total number of workers in cluster

        Returns:
            Configuration dict for this worker
        """
        return self.config.get_worker_config(worker_id, rank, world_size)

    def set_worker_batch_size(self, worker_id: str, batch_size: int):
        """
        Set custom batch size for a worker based on its hardware capabilities.

        Args:
            worker_id: Worker identifier
            batch_size: Batch size for this worker
        """
        self.config.set_worker_override(worker_id, 'batch_size', batch_size)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self.config.to_dict()
