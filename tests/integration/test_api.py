"""
Integration tests for coordinator API endpoints.

Tests the REST API and WebSocket functionality.
"""

import pytest
import os
import tempfile
from fastapi.testclient import TestClient

# Import after ensuring coordinator modules exist
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from coordinator.server import app
from coordinator.database import Database


@pytest.fixture
def client():
    """Create test client with temporary database."""
    # Create temporary database
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    # Override database path for testing
    original_db_path = "coordinator.db"

    # Import globals and override
    from coordinator import server
    server.db = Database(db_path)
    from coordinator.registry import WorkerRegistry
    from coordinator.clustering import ClusterManager
    server.registry = WorkerRegistry(server.db, heartbeat_timeout=90)
    server.cluster_manager = ClusterManager(latency_threshold_ms=50.0)

    # Create test client
    test_client = TestClient(app)

    yield test_client

    # Cleanup
    server.db.close()
    os.unlink(db_path)


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data['service'] == "Legion Coordinator"
        assert data['status'] == "running"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == "healthy"
        assert 'workers' in data


class TestWorkerEndpoints:
    """Test worker registration and management endpoints."""

    def test_register_worker(self, client):
        """Test worker registration."""
        response = client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051,
            "gpu_info": {"name": "RTX 3080", "memory_gb": 10},
            "cpu_cores": 8,
            "ram_gb": 32.0,
            "bandwidth_mbps": 1000.0
        })

        assert response.status_code == 201
        data = response.json()
        assert data['worker_id'] == "worker_1"

    def test_register_duplicate_worker(self, client):
        """Test registering duplicate worker."""
        # Register first time
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })

        # Try to register again
        response = client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })

        assert response.status_code == 409

    def test_get_worker(self, client):
        """Test getting worker information."""
        # Register worker
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051,
            "cpu_cores": 8
        })

        # Get worker
        response = client.get("/workers/worker_1")
        assert response.status_code == 200

        data = response.json()
        assert data['worker_id'] == "worker_1"
        assert data['ip_address'] == "192.168.1.100"
        assert data['cpu_cores'] == 8

    def test_get_nonexistent_worker(self, client):
        """Test getting nonexistent worker."""
        response = client.get("/workers/nonexistent")
        assert response.status_code == 404

    def test_list_workers(self, client):
        """Test listing workers."""
        # Register multiple workers
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })
        client.post("/workers/register", json={
            "worker_id": "worker_2",
            "ip_address": "192.168.1.101",
            "port": 50051
        })

        # List all workers
        response = client.get("/workers")
        assert response.status_code == 200

        data = response.json()
        assert data['count'] == 2
        assert len(data['workers']) == 2

    def test_list_workers_by_status(self, client):
        """Test listing workers filtered by status."""
        # Register workers
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })
        client.post("/workers/register", json={
            "worker_id": "worker_2",
            "ip_address": "192.168.1.101",
            "port": 50051
        })

        # List online workers
        response = client.get("/workers?status=online")
        assert response.status_code == 200

        data = response.json()
        assert data['count'] == 2

    def test_heartbeat(self, client):
        """Test worker heartbeat."""
        # Register worker
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })

        # Send heartbeat
        response = client.post("/workers/worker_1/heartbeat")
        assert response.status_code == 200

    def test_heartbeat_unknown_worker(self, client):
        """Test heartbeat from unknown worker."""
        response = client.post("/workers/unknown/heartbeat")
        assert response.status_code == 404

    def test_deregister_worker(self, client):
        """Test worker deregistration."""
        # Register worker
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })

        # Deregister
        response = client.delete("/workers/worker_1")
        assert response.status_code == 200

        # Verify worker is gone
        response = client.get("/workers/worker_1")
        assert response.status_code == 404

    def test_assign_shard(self, client):
        """Test shard assignment."""
        # Register worker
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })

        # Assign shard
        response = client.post("/workers/assign-shard", json={
            "worker_id": "worker_1",
            "shard_start": 0,
            "shard_end": 1000
        })

        assert response.status_code == 200

        # Verify assignment
        response = client.get("/workers/worker_1")
        data = response.json()
        assert data['shard_start'] == 0
        assert data['shard_end'] == 1000


class TestClusterEndpoints:
    """Test clustering endpoints."""

    def test_report_latency(self, client):
        """Test latency reporting."""
        response = client.post("/latency/report", json={
            "worker_a": "worker_1",
            "worker_b": "worker_2",
            "latency_ms": 25.5
        })

        assert response.status_code == 200

    def test_compute_clusters_no_workers(self, client):
        """Test cluster computation with no workers."""
        response = client.post("/clusters/compute")
        assert response.status_code == 200

        data = response.json()
        assert data['stats']['num_clusters'] == 0

    def test_compute_clusters(self, client):
        """Test cluster computation."""
        # Register workers
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })
        client.post("/workers/register", json={
            "worker_id": "worker_2",
            "ip_address": "192.168.1.101",
            "port": 50051
        })

        # Report latencies
        client.post("/latency/report", json={
            "worker_a": "worker_1",
            "worker_b": "worker_2",
            "latency_ms": 25.0
        })

        # Compute clusters
        response = client.post("/clusters/compute")
        assert response.status_code == 200

        data = response.json()
        assert data['stats']['num_clusters'] >= 1

    def test_get_clusters(self, client):
        """Test getting cluster assignments."""
        # Register workers
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })
        client.post("/workers/register", json={
            "worker_id": "worker_2",
            "ip_address": "192.168.1.101",
            "port": 50051
        })

        # Compute clusters
        client.post("/clusters/compute")

        # Get clusters
        response = client.get("/clusters")
        assert response.status_code == 200

        data = response.json()
        assert 'clusters' in data
        assert data['count'] >= 1


class TestMetricsEndpoints:
    """Test metrics endpoints."""

    def test_report_metric(self, client):
        """Test metric reporting."""
        # Register worker first
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })

        # Report metric
        response = client.post("/metrics/report", json={
            "worker_id": "worker_1",
            "global_step": 100,
            "loss": 2.5,
            "throughput": 50.0,
            "memory_usage_gb": 8.0
        })

        assert response.status_code == 200

    def test_get_metrics(self, client):
        """Test getting metrics."""
        # Register worker and report metric
        client.post("/workers/register", json={
            "worker_id": "worker_1",
            "ip_address": "192.168.1.100",
            "port": 50051
        })
        client.post("/metrics/report", json={
            "worker_id": "worker_1",
            "global_step": 100,
            "loss": 2.5
        })

        # Get metrics
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert data['count'] >= 1
        assert len(data['metrics']) >= 1

    def test_get_metrics_with_limit(self, client):
        """Test getting metrics with limit."""
        response = client.get("/metrics?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert len(data['metrics']) <= 10
