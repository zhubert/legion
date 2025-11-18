"""
Version management for asynchronous training.

Tracks worker training versions to:
- Compute global version (median of active workers)
- Enforce bounded staleness
- Coordinate work stealing assignments
- Identify slow/struggling workers
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class WorkerVersion:
    """Worker version information."""
    worker_id: str
    version: int
    timestamp: float
    is_healthy: bool


class VersionManager:
    """
    Manages training version tracking across distributed workers.

    Key responsibilities:
    - Track each worker's current training step
    - Compute global version (median of active workers)
    - Identify slow workers (below median)
    - Identify ahead workers (above median + staleness_bound)
    - Coordinate work stealing assignments
    """

    def __init__(self, staleness_bound: int = 5):
        """
        Initialize version manager.

        Args:
            staleness_bound: Maximum allowed version gap from global version
        """
        self.staleness_bound = staleness_bound
        self.worker_versions: Dict[str, WorkerVersion] = {}
        self.version_history: List[Tuple[float, int]] = []  # (timestamp, global_version)
        self.max_history = 1000

        # Work stealing assignment
        self.backup_assignments: Dict[str, str] = {}  # ahead_worker_id -> slow_worker_id

        logger.info(f"Initialized VersionManager (staleness_bound={staleness_bound})")

    def update_worker_version(self, worker_id: str, version: int, is_healthy: bool = True):
        """
        Update a worker's current version.

        Args:
            worker_id: Worker identifier
            version: Current training step version
            is_healthy: Whether worker is responding to heartbeats
        """
        self.worker_versions[worker_id] = WorkerVersion(
            worker_id=worker_id,
            version=version,
            timestamp=time.time(),
            is_healthy=is_healthy
        )

        logger.debug(f"Worker {worker_id} updated to version {version}")

    def mark_worker_offline(self, worker_id: str):
        """
        Mark a worker as offline (remove from version tracking).

        Args:
            worker_id: Worker identifier
        """
        if worker_id in self.worker_versions:
            del self.worker_versions[worker_id]
            logger.info(f"Removed worker {worker_id} from version tracking")

        # Clean up any backup assignments involving this worker
        if worker_id in self.backup_assignments:
            del self.backup_assignments[worker_id]

        # Remove as target of backup assignments
        self.backup_assignments = {
            k: v for k, v in self.backup_assignments.items()
            if v != worker_id
        }

    def get_global_version(self) -> int:
        """
        Compute global version as median of active healthy workers.

        Using median instead of minimum prevents one straggler from
        blocking the entire cluster.

        Returns:
            Global version (median of active workers), 0 if no workers
        """
        active_versions = [
            wv.version for wv in self.worker_versions.values()
            if wv.is_healthy
        ]

        if not active_versions:
            return 0

        global_version = int(statistics.median(active_versions))

        # Record in history
        self.version_history.append((time.time(), global_version))
        if len(self.version_history) > self.max_history:
            self.version_history = self.version_history[-self.max_history:]

        return global_version

    def get_slow_workers(self, global_version: Optional[int] = None) -> List[str]:
        """
        Get list of workers below global version.

        Args:
            global_version: Global version to compare against (computed if None)

        Returns:
            List of worker IDs below global version, sorted slowest first
        """
        if global_version is None:
            global_version = self.get_global_version()

        slow_workers = [
            (wv.worker_id, wv.version)
            for wv in self.worker_versions.values()
            if wv.is_healthy and wv.version < global_version
        ]

        # Sort by version (slowest first)
        slow_workers.sort(key=lambda x: x[1])

        return [worker_id for worker_id, _ in slow_workers]

    def get_ahead_workers(self, global_version: Optional[int] = None) -> List[str]:
        """
        Get list of workers ahead of global version + staleness_bound.

        These workers should do work stealing instead of progressing.

        Args:
            global_version: Global version to compare against (computed if None)

        Returns:
            List of worker IDs that are too far ahead
        """
        if global_version is None:
            global_version = self.get_global_version()

        threshold = global_version + self.staleness_bound

        ahead_workers = [
            wv.worker_id
            for wv in self.worker_versions.values()
            if wv.is_healthy and wv.version > threshold
        ]

        return ahead_workers

    def assign_work_stealing(self) -> Dict[str, str]:
        """
        Assign ahead workers to help slow workers.

        Returns:
            Dict mapping ahead_worker_id -> slow_worker_id for backup computation
        """
        global_version = self.get_global_version()
        slow_workers = self.get_slow_workers(global_version)
        ahead_workers = self.get_ahead_workers(global_version)

        # Clear old assignments
        self.backup_assignments = {}

        # Assign ahead workers to help slowest workers
        for i, ahead_worker in enumerate(ahead_workers):
            if i < len(slow_workers):
                slow_worker = slow_workers[i]
                self.backup_assignments[ahead_worker] = slow_worker
                logger.info(
                    f"Assigned {ahead_worker} to help {slow_worker} "
                    f"(ahead: {self.worker_versions[ahead_worker].version}, "
                    f"slow: {self.worker_versions[slow_worker].version})"
                )

        return self.backup_assignments

    def get_backup_assignment(self, worker_id: str) -> Optional[str]:
        """
        Get work stealing assignment for a worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Slow worker ID to help, or None if no assignment
        """
        return self.backup_assignments.get(worker_id)

    def get_worker_info(self, worker_id: str) -> Optional[WorkerVersion]:
        """
        Get version info for a specific worker.

        Args:
            worker_id: Worker identifier

        Returns:
            WorkerVersion or None if worker not tracked
        """
        return self.worker_versions.get(worker_id)

    def get_all_workers(self) -> List[WorkerVersion]:
        """
        Get all tracked worker versions.

        Returns:
            List of WorkerVersion objects
        """
        return list(self.worker_versions.values())

    def get_staleness_stats(self) -> Dict[str, float]:
        """
        Compute staleness statistics.

        Returns:
            Dict with min, max, mean, median version and staleness metrics
        """
        if not self.worker_versions:
            return {
                "num_workers": 0,
                "global_version": 0,
                "min_version": 0,
                "max_version": 0,
                "mean_version": 0,
                "median_version": 0,
                "version_range": 0,
                "staleness_bound": self.staleness_bound
            }

        versions = [wv.version for wv in self.worker_versions.values() if wv.is_healthy]

        return {
            "num_workers": len(versions),
            "global_version": self.get_global_version(),
            "min_version": min(versions),
            "max_version": max(versions),
            "mean_version": statistics.mean(versions),
            "median_version": statistics.median(versions),
            "version_range": max(versions) - min(versions),
            "staleness_bound": self.staleness_bound
        }

    def is_worker_too_far_ahead(self, worker_id: str) -> bool:
        """
        Check if a worker is beyond the staleness bound.

        Args:
            worker_id: Worker identifier

        Returns:
            True if worker should stop and do work stealing
        """
        if worker_id not in self.worker_versions:
            return False

        worker_version = self.worker_versions[worker_id].version
        global_version = self.get_global_version()

        return worker_version > global_version + self.staleness_bound

    def is_worker_too_far_behind(self, worker_id: str) -> bool:
        """
        Check if a worker is lagging behind the cluster.

        Args:
            worker_id: Worker identifier

        Returns:
            True if worker is more than staleness_bound behind
        """
        if worker_id not in self.worker_versions:
            return False

        worker_version = self.worker_versions[worker_id].version
        global_version = self.get_global_version()

        return worker_version < global_version - self.staleness_bound

    def set_staleness_bound(self, staleness_bound: int):
        """
        Update the staleness bound.

        Args:
            staleness_bound: New staleness bound
        """
        if staleness_bound < 1:
            raise ValueError(f"Staleness bound must be >= 1, got {staleness_bound}")

        self.staleness_bound = staleness_bound
        logger.info(f"Updated staleness bound to {staleness_bound}")

    def get_version_progress_rate(self, window_seconds: float = 60.0) -> float:
        """
        Compute cluster's version progress rate (versions/second).

        Args:
            window_seconds: Time window to measure over

        Returns:
            Average versions per second, or 0 if insufficient data
        """
        if len(self.version_history) < 2:
            return 0.0

        # Filter to recent history
        cutoff_time = time.time() - window_seconds
        recent_history = [
            (ts, v) for ts, v in self.version_history
            if ts >= cutoff_time
        ]

        if len(recent_history) < 2:
            return 0.0

        # Compute rate
        earliest_ts, earliest_version = recent_history[0]
        latest_ts, latest_version = recent_history[-1]

        time_delta = latest_ts - earliest_ts
        version_delta = latest_version - earliest_version

        if time_delta == 0:
            return 0.0

        return version_delta / time_delta
