"""
Worker registry management for the coordinator.

Provides high-level operations for managing worker lifecycle:
- Registration and discovery
- Heartbeat monitoring
- Health checks
- Shard assignment
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from coordinator.database import Database


logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Worker metadata."""
    worker_id: str
    ip_address: str
    port: int
    status: str
    gpu_info: Optional[Dict[str, Any]] = None
    cpu_cores: Optional[int] = None
    ram_gb: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    region: Optional[str] = None
    shard_start: Optional[int] = None
    shard_end: Optional[int] = None
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerInfo':
        """Create WorkerInfo from database row dictionary."""
        return cls(
            worker_id=data['worker_id'],
            ip_address=data['ip_address'],
            port=data['port'],
            status=data['status'],
            gpu_info=data.get('gpu_info'),
            cpu_cores=data.get('cpu_cores'),
            ram_gb=data.get('ram_gb'),
            bandwidth_mbps=data.get('bandwidth_mbps'),
            region=data.get('region'),
            shard_start=data.get('shard_start'),
            shard_end=data.get('shard_end'),
            registered_at=data.get('registered_at'),
            last_heartbeat=data.get('last_heartbeat')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        # Handle both datetime objects and strings from database
        registered_at_str = None
        if self.registered_at:
            if isinstance(self.registered_at, str):
                registered_at_str = self.registered_at
            else:
                registered_at_str = self.registered_at.isoformat()

        last_heartbeat_str = None
        if self.last_heartbeat:
            if isinstance(self.last_heartbeat, str):
                last_heartbeat_str = self.last_heartbeat
            else:
                last_heartbeat_str = self.last_heartbeat.isoformat()

        return {
            'worker_id': self.worker_id,
            'ip_address': self.ip_address,
            'port': self.port,
            'status': self.status,
            'gpu_info': self.gpu_info,
            'cpu_cores': self.cpu_cores,
            'ram_gb': self.ram_gb,
            'bandwidth_mbps': self.bandwidth_mbps,
            'region': self.region,
            'shard_start': self.shard_start,
            'shard_end': self.shard_end,
            'registered_at': registered_at_str,
            'last_heartbeat': last_heartbeat_str
        }


class WorkerRegistry:
    """
    Worker registry with health monitoring.

    Manages the lifecycle of workers in the distributed training system.
    """

    def __init__(self, db: Database, heartbeat_timeout: int = 90):
        """
        Initialize worker registry.

        Args:
            db: Database instance
            heartbeat_timeout: Heartbeat timeout in seconds (default: 90s)
        """
        self.db = db
        self.heartbeat_timeout = heartbeat_timeout

    def register_worker(
        self,
        worker_id: str,
        ip_address: str,
        port: int,
        gpu_info: Optional[Dict[str, Any]] = None,
        cpu_cores: Optional[int] = None,
        ram_gb: Optional[float] = None,
        bandwidth_mbps: Optional[float] = None
    ) -> bool:
        """
        Register a new worker.

        Args:
            worker_id: Unique worker identifier
            ip_address: Worker IP address
            port: Worker gRPC port
            gpu_info: GPU information (name, memory_gb, etc.)
            cpu_cores: Number of CPU cores
            ram_gb: RAM in GB
            bandwidth_mbps: Network bandwidth in Mbps

        Returns:
            True if registration successful, False if worker already exists
        """
        success = self.db.register_worker(
            worker_id=worker_id,
            ip_address=ip_address,
            port=port,
            gpu_info=gpu_info,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            bandwidth_mbps=bandwidth_mbps
        )

        if success:
            logger.info(f"Worker registered: {worker_id} at {ip_address}:{port}")
        else:
            logger.warning(f"Worker already exists: {worker_id}")

        return success

    def deregister_worker(self, worker_id: str) -> bool:
        """
        Deregister a worker (graceful shutdown).

        Args:
            worker_id: Worker identifier

        Returns:
            True if deregistration successful
        """
        success = self.db.deregister_worker(worker_id)

        if success:
            logger.info(f"Worker deregistered: {worker_id}")
        else:
            logger.warning(f"Worker not found: {worker_id}")

        return success

    def heartbeat(self, worker_id: str) -> bool:
        """
        Process worker heartbeat.

        Updates the last heartbeat timestamp and marks worker as online.

        Args:
            worker_id: Worker identifier

        Returns:
            True if heartbeat successful, False if worker not found
        """
        success = self.db.update_heartbeat(worker_id)

        if not success:
            logger.warning(f"Heartbeat from unknown worker: {worker_id}")

        return success

    def check_stale_workers(self) -> List[WorkerInfo]:
        """
        Find workers with stale heartbeats and mark them offline.

        Returns:
            List of workers marked offline
        """
        stale_workers = self.db.get_stale_workers(self.heartbeat_timeout)

        offline_workers = []
        for worker_data in stale_workers:
            worker_id = worker_data['worker_id']
            self.db.mark_worker_offline(worker_id)
            logger.warning(f"Worker marked offline (stale heartbeat): {worker_id}")
            offline_workers.append(WorkerInfo.from_dict(worker_data))

        return offline_workers

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """
        Get worker information.

        Args:
            worker_id: Worker identifier

        Returns:
            WorkerInfo or None if not found
        """
        worker_data = self.db.get_worker(worker_id)

        if worker_data is None:
            return None

        return WorkerInfo.from_dict(worker_data)

    def get_all_workers(self, status: Optional[str] = None) -> List[WorkerInfo]:
        """
        Get all workers, optionally filtered by status.

        Args:
            status: Filter by status ('online', 'offline', None for all)

        Returns:
            List of WorkerInfo
        """
        workers_data = self.db.get_all_workers(status=status)
        return [WorkerInfo.from_dict(w) for w in workers_data]

    def get_online_workers(self) -> List[WorkerInfo]:
        """
        Get all online workers.

        Returns:
            List of online WorkerInfo
        """
        return self.get_all_workers(status='online')

    def assign_shard(self, worker_id: str, shard_start: int, shard_end: int) -> bool:
        """
        Assign parameter shard to worker.

        Args:
            worker_id: Worker identifier
            shard_start: Start index of parameter shard
            shard_end: End index of parameter shard

        Returns:
            True if assignment successful
        """
        success = self.db.assign_shard(worker_id, shard_start, shard_end)

        if success:
            logger.info(f"Assigned shard [{shard_start}:{shard_end}] to worker {worker_id}")
        else:
            logger.warning(f"Failed to assign shard to worker {worker_id}")

        return success

    def assign_region(self, worker_id: str, region: str) -> bool:
        """
        Assign worker to regional cluster.

        Args:
            worker_id: Worker identifier
            region: Region name

        Returns:
            True if assignment successful
        """
        success = self.db.assign_region(worker_id, region)

        if success:
            logger.info(f"Assigned worker {worker_id} to region {region}")
        else:
            logger.warning(f"Failed to assign region to worker {worker_id}")

        return success

    def get_worker_count(self) -> Dict[str, int]:
        """
        Get worker counts by status.

        Returns:
            Dictionary with counts: {'total': N, 'online': M, 'offline': K}
        """
        all_workers = self.get_all_workers()
        online_workers = [w for w in all_workers if w.status == 'online']
        offline_workers = [w for w in all_workers if w.status == 'offline']

        return {
            'total': len(all_workers),
            'online': len(online_workers),
            'offline': len(offline_workers)
        }

    def get_workers_by_region(self) -> Dict[str, List[WorkerInfo]]:
        """
        Group workers by region.

        Returns:
            Dictionary mapping region name to list of workers
        """
        workers = self.get_online_workers()
        regions: Dict[str, List[WorkerInfo]] = {}

        for worker in workers:
            region = worker.region or 'unassigned'
            if region not in regions:
                regions[region] = []
            regions[region].append(worker)

        return regions
