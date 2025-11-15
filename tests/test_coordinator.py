"""
Unit tests for coordinator components.

Tests:
- Database operations
- Worker registry
- Regional clustering
- API endpoints
"""

import pytest
import os
import tempfile
from datetime import datetime
import time

from coordinator.database import Database
from coordinator.registry import WorkerRegistry, WorkerInfo
from coordinator.clustering import ClusterManager, Cluster


class TestDatabase:
    """Test database operations."""

    @pytest.fixture
    def db(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)

        database = Database(path)
        yield database

        database.close()
        os.unlink(path)

    def test_register_worker(self, db):
        """Test worker registration."""
        success = db.register_worker(
            worker_id="worker_1",
            ip_address="192.168.1.100",
            port=50051,
            gpu_info={"name": "RTX 4090", "memory_gb": 24},
            cpu_cores=16,
            ram_gb=64.0,
            bandwidth_mbps=1000.0
        )

        assert success is True

        # Try to register same worker again
        success = db.register_worker(
            worker_id="worker_1",
            ip_address="192.168.1.100",
            port=50051
        )

        assert success is False

    def test_get_worker(self, db):
        """Test retrieving worker information."""
        db.register_worker(
            worker_id="worker_1",
            ip_address="192.168.1.100",
            port=50051,
            cpu_cores=8
        )

        worker = db.get_worker("worker_1")

        assert worker is not None
        assert worker['worker_id'] == "worker_1"
        assert worker['ip_address'] == "192.168.1.100"
        assert worker['port'] == 50051
        assert worker['cpu_cores'] == 8
        assert worker['status'] == 'online'

    def test_update_heartbeat(self, db):
        """Test heartbeat update."""
        db.register_worker(
            worker_id="worker_1",
            ip_address="192.168.1.100",
            port=50051
        )

        time.sleep(0.1)

        success = db.update_heartbeat("worker_1")
        assert success is True

        worker = db.get_worker("worker_1")
        assert worker['status'] == 'online'

    def test_deregister_worker(self, db):
        """Test worker deregistration."""
        db.register_worker(
            worker_id="worker_1",
            ip_address="192.168.1.100",
            port=50051
        )

        success = db.deregister_worker("worker_1")
        assert success is True

        worker = db.get_worker("worker_1")
        assert worker is None

    def test_get_all_workers(self, db):
        """Test getting all workers."""
        db.register_worker("worker_1", "192.168.1.100", 50051)
        db.register_worker("worker_2", "192.168.1.101", 50051)
        db.register_worker("worker_3", "192.168.1.102", 50051)

        workers = db.get_all_workers()
        assert len(workers) == 3

        # Mark one offline
        db.mark_worker_offline("worker_2")

        online_workers = db.get_all_workers(status='online')
        assert len(online_workers) == 2

        offline_workers = db.get_all_workers(status='offline')
        assert len(offline_workers) == 1

    def test_assign_shard(self, db):
        """Test shard assignment."""
        db.register_worker("worker_1", "192.168.1.100", 50051)

        success = db.assign_shard("worker_1", 0, 1000)
        assert success is True

        worker = db.get_worker("worker_1")
        assert worker['shard_start'] == 0
        assert worker['shard_end'] == 1000

    def test_assign_region(self, db):
        """Test region assignment."""
        db.register_worker("worker_1", "192.168.1.100", 50051)

        success = db.assign_region("worker_1", "us-west")
        assert success is True

        worker = db.get_worker("worker_1")
        assert worker['region'] == "us-west"

    def test_record_metric(self, db):
        """Test metric recording."""
        db.register_worker("worker_1", "192.168.1.100", 50051)

        success = db.record_metric(
            worker_id="worker_1",
            global_step=100,
            loss=2.5,
            throughput=50.0,
            memory_usage_gb=8.0
        )

        assert success is True

        metrics = db.get_recent_metrics(limit=10)
        assert len(metrics) > 0
        assert metrics[0]['worker_id'] == "worker_1"
        assert metrics[0]['loss'] == 2.5


class TestWorkerRegistry:
    """Test worker registry."""

    @pytest.fixture
    def registry(self):
        """Create registry with temporary database."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)

        db = Database(path)
        reg = WorkerRegistry(db, heartbeat_timeout=1)  # 1 second timeout for testing

        yield reg

        db.close()
        os.unlink(path)

    def test_register_worker(self, registry):
        """Test worker registration."""
        success = registry.register_worker(
            worker_id="worker_1",
            ip_address="192.168.1.100",
            port=50051,
            gpu_info={"name": "RTX 3080"},
            cpu_cores=8
        )

        assert success is True

        worker = registry.get_worker("worker_1")
        assert worker is not None
        assert worker.worker_id == "worker_1"
        assert worker.ip_address == "192.168.1.100"

    def test_heartbeat(self, registry):
        """Test heartbeat processing."""
        registry.register_worker("worker_1", "192.168.1.100", 50051)

        success = registry.heartbeat("worker_1")
        assert success is True

        # Unknown worker
        success = registry.heartbeat("worker_unknown")
        assert success is False

    def test_check_stale_workers(self, registry):
        """Test stale worker detection."""
        registry.register_worker("worker_1", "192.168.1.100", 50051)
        registry.register_worker("worker_2", "192.168.1.101", 50051)

        # Wait for timeout
        time.sleep(1.5)

        stale_workers = registry.check_stale_workers()
        assert len(stale_workers) == 2

        # Workers should be marked offline
        worker1 = registry.get_worker("worker_1")
        assert worker1.status == 'offline'

    def test_get_online_workers(self, registry):
        """Test getting online workers."""
        registry.register_worker("worker_1", "192.168.1.100", 50051)
        registry.register_worker("worker_2", "192.168.1.101", 50051)
        registry.register_worker("worker_3", "192.168.1.102", 50051)

        # Mark one offline
        registry.db.mark_worker_offline("worker_2")

        online_workers = registry.get_online_workers()
        assert len(online_workers) == 2

    def test_worker_count(self, registry):
        """Test worker count statistics."""
        registry.register_worker("worker_1", "192.168.1.100", 50051)
        registry.register_worker("worker_2", "192.168.1.101", 50051)

        counts = registry.get_worker_count()
        assert counts['total'] == 2
        assert counts['online'] == 2
        assert counts['offline'] == 0

        # Mark one offline
        registry.db.mark_worker_offline("worker_1")

        counts = registry.get_worker_count()
        assert counts['online'] == 1
        assert counts['offline'] == 1

    def test_workers_by_region(self, registry):
        """Test grouping workers by region."""
        registry.register_worker("worker_1", "192.168.1.100", 50051)
        registry.register_worker("worker_2", "192.168.1.101", 50051)
        registry.register_worker("worker_3", "192.168.1.102", 50051)

        registry.assign_region("worker_1", "us-west")
        registry.assign_region("worker_2", "us-west")
        registry.assign_region("worker_3", "us-east")

        regions = registry.get_workers_by_region()

        assert len(regions) == 2
        assert len(regions['us-west']) == 2
        assert len(regions['us-east']) == 1


class TestClusterManager:
    """Test regional clustering."""

    def test_update_latency(self):
        """Test latency measurement update."""
        manager = ClusterManager(latency_threshold_ms=50.0)

        manager.update_latency("worker_1", "worker_2", 25.0)

        latency = manager.get_latency("worker_1", "worker_2")
        assert latency == 25.0

        # Should be symmetric
        latency = manager.get_latency("worker_2", "worker_1")
        assert latency == 25.0

    def test_simple_clustering(self):
        """Test simple cluster assignment."""
        manager = ClusterManager()

        workers = [
            WorkerInfo("worker_1", "192.168.1.100", 50051, "online"),
            WorkerInfo("worker_2", "192.168.1.101", 50051, "online"),
            WorkerInfo("worker_3", "192.168.1.102", 50051, "online")
        ]

        assignments = manager.assign_clusters_simple(workers)

        assert len(assignments) == 3
        assert all(region == "global" for region in assignments.values())

    def test_latency_based_clustering(self):
        """Test clustering based on latency measurements."""
        manager = ClusterManager(latency_threshold_ms=50.0)

        # Create workers
        workers = [
            WorkerInfo("worker_1", "192.168.1.100", 50051, "online"),
            WorkerInfo("worker_2", "192.168.1.101", 50051, "online"),
            WorkerInfo("worker_3", "192.168.1.102", 50051, "online"),
            WorkerInfo("worker_4", "192.168.1.103", 50051, "online")
        ]

        # Set up latencies
        # worker_1 and worker_2 are close (same cluster)
        manager.update_latency("worker_1", "worker_2", 20.0)
        # worker_3 and worker_4 are close (same cluster)
        manager.update_latency("worker_3", "worker_4", 25.0)
        # But worker_1/2 are far from worker_3/4 (different clusters)
        manager.update_latency("worker_1", "worker_3", 100.0)
        manager.update_latency("worker_1", "worker_4", 100.0)
        manager.update_latency("worker_2", "worker_3", 100.0)
        manager.update_latency("worker_2", "worker_4", 100.0)

        clusters = manager.compute_clusters(workers)

        # Should create 2 clusters
        assert len(clusters) == 2

        # Each cluster should have 2 workers
        assert all(len(c.worker_ids) == 2 for c in clusters)

    def test_single_cluster(self):
        """Test all workers in single cluster."""
        manager = ClusterManager(latency_threshold_ms=50.0)

        workers = [
            WorkerInfo("worker_1", "192.168.1.100", 50051, "online"),
            WorkerInfo("worker_2", "192.168.1.101", 50051, "online"),
            WorkerInfo("worker_3", "192.168.1.102", 50051, "online")
        ]

        # All workers close to each other
        manager.update_latency("worker_1", "worker_2", 20.0)
        manager.update_latency("worker_1", "worker_3", 25.0)
        manager.update_latency("worker_2", "worker_3", 30.0)

        clusters = manager.compute_clusters(workers)

        # Should create 1 cluster
        assert len(clusters) == 1
        assert len(clusters[0].worker_ids) == 3

    def test_cluster_stats(self):
        """Test cluster statistics."""
        manager = ClusterManager()

        clusters = [
            Cluster("cluster_0", "region_0", ["w1", "w2", "w3"], 25.0),
            Cluster("cluster_1", "region_1", ["w4", "w5"], 30.0)
        ]

        stats = manager.get_cluster_stats(clusters)

        assert stats['num_clusters'] == 2
        assert stats['total_workers'] == 5
        assert stats['avg_cluster_size'] == 2.5
        assert stats['avg_intra_cluster_latency'] == 27.5

    def test_empty_clusters(self):
        """Test stats with no clusters."""
        manager = ClusterManager()

        stats = manager.get_cluster_stats([])

        assert stats['num_clusters'] == 0
        assert stats['total_workers'] == 0
        assert stats['avg_cluster_size'] == 0.0
