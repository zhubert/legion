"""
Database module for coordinator state management.

Uses SQLite for persistent storage of worker metadata, cluster assignments,
and checkpoint information.
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import threading


class Database:
    """
    Thread-safe SQLite database for coordinator state.

    Stores:
    - Worker registry (ID, IP, GPU info, status)
    - Regional clusters
    - Checkpoint metadata
    - Training metrics
    """

    def __init__(self, db_path: str = "coordinator.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Workers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                ip_address TEXT NOT NULL,
                port INTEGER NOT NULL,
                gpu_info TEXT,
                cpu_cores INTEGER,
                ram_gb REAL,
                bandwidth_mbps REAL,
                status TEXT NOT NULL,
                region TEXT,
                shard_start INTEGER,
                shard_end INTEGER,
                registered_at TIMESTAMP NOT NULL,
                last_heartbeat TIMESTAMP NOT NULL
            )
        """)

        # Clusters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                region TEXT NOT NULL,
                worker_ids TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                global_step INTEGER NOT NULL,
                worker_shards TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP NOT NULL
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT NOT NULL,
                global_step INTEGER NOT NULL,
                loss REAL,
                throughput REAL,
                memory_usage_gb REAL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
            )
        """)

        conn.commit()

    # Worker operations

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
            port: Worker port
            gpu_info: GPU information (name, memory, etc.)
            cpu_cores: Number of CPU cores
            ram_gb: RAM in GB
            bandwidth_mbps: Network bandwidth in Mbps

        Returns:
            True if registration successful, False if worker already exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow()

        try:
            cursor.execute("""
                INSERT INTO workers (
                    worker_id, ip_address, port, gpu_info, cpu_cores, ram_gb,
                    bandwidth_mbps, status, registered_at, last_heartbeat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                worker_id, ip_address, port,
                json.dumps(gpu_info) if gpu_info else None,
                cpu_cores, ram_gb, bandwidth_mbps,
                'online', now, now
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Worker already exists
            return False

    def update_heartbeat(self, worker_id: str) -> bool:
        """
        Update worker's last heartbeat timestamp.

        Args:
            worker_id: Worker identifier

        Returns:
            True if update successful, False if worker not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE workers
            SET last_heartbeat = ?, status = 'online'
            WHERE worker_id = ?
        """, (datetime.utcnow(), worker_id))

        conn.commit()
        return cursor.rowcount > 0

    def deregister_worker(self, worker_id: str) -> bool:
        """
        Remove worker from registry.

        Args:
            worker_id: Worker identifier

        Returns:
            True if removal successful, False if worker not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
        conn.commit()
        return cursor.rowcount > 0

    def mark_worker_offline(self, worker_id: str) -> bool:
        """
        Mark worker as offline (failed heartbeat).

        Args:
            worker_id: Worker identifier

        Returns:
            True if update successful, False if worker not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE workers
            SET status = 'offline'
            WHERE worker_id = ?
        """, (worker_id,))

        conn.commit()
        return cursor.rowcount > 0

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get worker information.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker info dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_all_workers(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all workers, optionally filtered by status.

        Args:
            status: Filter by status ('online', 'offline', None for all)

        Returns:
            List of worker info dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("SELECT * FROM workers WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT * FROM workers")

        return [dict(row) for row in cursor.fetchall()]

    def get_stale_workers(self, timeout_seconds: int = 90) -> List[Dict[str, Any]]:
        """
        Get workers with stale heartbeats.

        Args:
            timeout_seconds: Heartbeat timeout in seconds

        Returns:
            List of worker info dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM workers
            WHERE status = 'online'
            AND (julianday('now') - julianday(last_heartbeat)) * 86400 > ?
        """, (timeout_seconds,))

        return [dict(row) for row in cursor.fetchall()]

    def assign_shard(self, worker_id: str, shard_start: int, shard_end: int) -> bool:
        """
        Assign parameter shard to worker.

        Args:
            worker_id: Worker identifier
            shard_start: Start index of parameter shard
            shard_end: End index of parameter shard

        Returns:
            True if assignment successful, False if worker not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE workers
            SET shard_start = ?, shard_end = ?
            WHERE worker_id = ?
        """, (shard_start, shard_end, worker_id))

        conn.commit()
        return cursor.rowcount > 0

    def assign_region(self, worker_id: str, region: str) -> bool:
        """
        Assign worker to a regional cluster.

        Args:
            worker_id: Worker identifier
            region: Region name

        Returns:
            True if assignment successful, False if worker not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE workers
            SET region = ?
            WHERE worker_id = ?
        """, (region, worker_id))

        conn.commit()
        return cursor.rowcount > 0

    # Cluster operations

    def create_cluster(self, cluster_id: str, region: str, worker_ids: List[str]) -> bool:
        """
        Create a regional cluster.

        Args:
            cluster_id: Unique cluster identifier
            region: Region name
            worker_ids: List of worker IDs in cluster

        Returns:
            True if creation successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow()

        try:
            cursor.execute("""
                INSERT INTO clusters (cluster_id, region, worker_ids, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (cluster_id, region, json.dumps(worker_ids), now, now))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def update_cluster(self, cluster_id: str, worker_ids: List[str]) -> bool:
        """
        Update cluster membership.

        Args:
            cluster_id: Cluster identifier
            worker_ids: Updated list of worker IDs

        Returns:
            True if update successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE clusters
            SET worker_ids = ?, updated_at = ?
            WHERE cluster_id = ?
        """, (json.dumps(worker_ids), datetime.utcnow(), cluster_id))

        conn.commit()
        return cursor.rowcount > 0

    def get_clusters(self) -> List[Dict[str, Any]]:
        """
        Get all clusters.

        Returns:
            List of cluster info dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM clusters")

        clusters = []
        for row in cursor.fetchall():
            cluster = dict(row)
            cluster['worker_ids'] = json.loads(cluster['worker_ids'])
            clusters.append(cluster)

        return clusters

    # Checkpoint operations

    def save_checkpoint_metadata(
        self,
        checkpoint_id: str,
        version: int,
        global_step: int,
        worker_shards: Dict[str, Dict[str, int]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save checkpoint metadata.

        Args:
            checkpoint_id: Unique checkpoint identifier
            version: Checkpoint version
            global_step: Training step
            worker_shards: Map of worker_id to shard info
            metadata: Additional metadata

        Returns:
            True if save successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO checkpoints (
                    checkpoint_id, version, global_step, worker_shards, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id, version, global_step,
                json.dumps(worker_shards),
                json.dumps(metadata) if metadata else None,
                datetime.utcnow()
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get most recent checkpoint metadata.

        Returns:
            Checkpoint info dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM checkpoints
            ORDER BY version DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if row is None:
            return None

        checkpoint = dict(row)
        checkpoint['worker_shards'] = json.loads(checkpoint['worker_shards'])
        if checkpoint['metadata']:
            checkpoint['metadata'] = json.loads(checkpoint['metadata'])

        return checkpoint

    # Metrics operations

    def record_metric(
        self,
        worker_id: str,
        global_step: int,
        loss: Optional[float] = None,
        throughput: Optional[float] = None,
        memory_usage_gb: Optional[float] = None
    ) -> bool:
        """
        Record training metric.

        Args:
            worker_id: Worker identifier
            global_step: Training step
            loss: Loss value
            throughput: Throughput (samples/sec)
            memory_usage_gb: Memory usage in GB

        Returns:
            True if recording successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO metrics (
                worker_id, global_step, loss, throughput, memory_usage_gb, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (worker_id, global_step, loss, throughput, memory_usage_gb, datetime.utcnow()))

        conn.commit()
        return True

    def record_metrics_batch(
        self,
        worker_id: str,
        metrics: List[Dict[str, Any]]
    ) -> bool:
        """
        Record multiple training metrics in a single transaction.

        Args:
            worker_id: Worker identifier
            metrics: List of metric dictionaries with keys:
                     - global_step (required)
                     - loss (optional)
                     - throughput (optional)
                     - memory_usage_gb (optional)

        Returns:
            True if recording successful
        """
        if not metrics:
            return True

        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow()

        # Prepare batch insert data
        batch_data = [
            (
                worker_id,
                metric['global_step'],
                metric.get('loss'),
                metric.get('throughput'),
                metric.get('memory_usage_gb'),
                now
            )
            for metric in metrics
        ]

        cursor.executemany("""
            INSERT INTO metrics (
                worker_id, global_step, loss, throughput, memory_usage_gb, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, batch_data)

        conn.commit()
        return True

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics across all workers.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of metric dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM metrics
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
