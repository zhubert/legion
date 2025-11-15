"""
Worker configuration for Legion distributed training.

Defines all configuration parameters for worker nodes.
"""

import os
import socket
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


def get_local_ip() -> str:
    """
    Get local IP address.

    Returns:
        Local IP address as string
    """
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_hostname() -> str:
    """
    Get hostname.

    Returns:
        Hostname as string
    """
    return socket.gethostname()


@dataclass
class WorkerConfig:
    """
    Configuration for a Legion worker node.

    This includes identity, network settings, hardware info,
    training parameters, and operational settings.
    """

    # Identity
    worker_id: str = field(
        default_factory=lambda: f"worker_{uuid.uuid4().hex[:8]}"
    )
    hostname: str = field(default_factory=get_hostname)

    # Coordinator connection
    coordinator_url: str = "http://localhost:8000"
    coordinator_timeout: float = 30.0  # seconds
    coordinator_retry_attempts: int = 3
    coordinator_retry_delay: float = 1.0  # seconds

    # Network settings
    ip_address: str = field(default_factory=get_local_ip)
    port: int = 50051  # gRPC port for worker-to-worker (Phase 1.3)

    # Hardware information
    device: str = "cpu"  # "cpu" or "cuda"
    cpu_cores: Optional[int] = None
    ram_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    bandwidth_mbps: Optional[float] = None

    # Training parameters
    model_size: str = "tiny"  # "tiny", "small", "medium"
    batch_size: int = 4
    seq_len: int = 32
    learning_rate: float = 0.001
    num_steps: int = 100

    # Compression
    compression: str = "none"  # "none", "int8", "topk"

    # Heartbeat settings
    heartbeat_interval: int = 30  # seconds between heartbeats
    heartbeat_timeout: int = 90  # coordinator timeout threshold
    heartbeat_max_failures: int = 3  # max consecutive failures before alerting

    # Telemetry settings
    telemetry_enabled: bool = True
    telemetry_interval_steps: int = 10  # report every N steps
    telemetry_buffer_size: int = 100  # max buffered metrics

    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval_steps: int = 100
    checkpoint_keep_last_n: int = 3  # keep only last N checkpoints

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_file: Optional[str] = None

    def __post_init__(self):
        """Validate and auto-detect hardware info."""
        # Auto-detect hardware if not specified
        if self.cpu_cores is None:
            self.cpu_cores = os.cpu_count() or 1

        if self.ram_gb is None:
            try:
                import psutil
                self.ram_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                self.ram_gb = None

        # Auto-detect GPU if device is cuda
        if self.device == "cuda" and (self.gpu_name is None or self.gpu_memory_gb is None):
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_name = torch.cuda.get_device_name(0)
                    self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    # Fall back to CPU if CUDA not available
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        # Create checkpoint directory if needed
        if self.checkpoint_enabled:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            'worker_id': self.worker_id,
            'hostname': self.hostname,
            'coordinator_url': self.coordinator_url,
            'ip_address': self.ip_address,
            'port': self.port,
            'device': self.device,
            'cpu_cores': self.cpu_cores,
            'ram_gb': self.ram_gb,
            'gpu_name': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory_gb,
            'bandwidth_mbps': self.bandwidth_mbps,
            'model_size': self.model_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'heartbeat_interval': self.heartbeat_interval,
            'telemetry_enabled': self.telemetry_enabled,
            'checkpoint_enabled': self.checkpoint_enabled
        }

    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """
        Get GPU information dictionary for registration.

        Returns:
            GPU info dict or None if no GPU
        """
        if self.device == "cuda" and self.gpu_name:
            return {
                'name': self.gpu_name,
                'memory_gb': self.gpu_memory_gb
            }
        return None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WorkerConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            WorkerConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, path: str) -> 'WorkerConfig':
        """
        Load config from JSON file.

        Args:
            path: Path to JSON config file

        Returns:
            WorkerConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json_file(self, path: str):
        """
        Save config to JSON file.

        Args:
            path: Path to save JSON config
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WorkerConfig(worker_id='{self.worker_id}', "
            f"coordinator='{self.coordinator_url}', "
            f"device='{self.device}', "
            f"model='{self.model_size}')"
        )
