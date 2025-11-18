"""
Tests for coordinator version management REST endpoints.

Tests the 7 version management endpoints added for async parameter server:
- POST /training/version/update
- GET /training/version/global
- GET /training/version/stats
- GET /training/version/workers
- GET /training/version/slow-workers
- GET /training/version/ahead-workers
- POST /training/version/assign-backups
"""

import pytest
import os
import tempfile
from fastapi.testclient import TestClient

from coordinator.server import app
from coordinator.database import Database
import coordinator.server as server_module


@pytest.fixture
def client():
    """Create test client with temporary database."""
    # Create temporary database
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    # Import and override globals
    from coordinator.registry import WorkerRegistry
    from coordinator.clustering import ClusterManager
    from coordinator.version_manager import VersionManager

    server_module.db = Database(db_path)
    server_module.registry = WorkerRegistry(server_module.db, heartbeat_timeout=90)
    server_module.cluster_manager = ClusterManager(latency_threshold_ms=50.0)
    server_module.version_manager = VersionManager(staleness_bound=5)

    # Create test client
    test_client = TestClient(app)

    yield test_client

    # Cleanup
    server_module.db.close()
    os.unlink(db_path)


class TestVersionEndpoints:
    """Test version management endpoints."""

    def test_update_worker_version(self, client):
        """Test POST /training/version/update endpoint."""
        # Update worker version
        response = client.post(
            "/training/version/update",
            json={
                "worker_id": "worker_0",
                "version": 10,
                "is_healthy": True
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["worker_id"] == "worker_0"
        assert data["version"] == 10

    def test_get_global_version(self, client):
        """Test GET /training/version/global endpoint."""
        # Add some workers
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_0", "version": 10, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_1", "version": 12, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_2", "version": 8, "is_healthy": True}
        )

        # Get global version
        response = client.get("/training/version/global")
        assert response.status_code == 200
        data = response.json()
        assert data["global_version"] == 10  # Median of [8, 10, 12]

    def test_get_version_stats(self, client):
        """Test GET /training/version/stats endpoint."""
        # Clear any previous state and add workers
        server_module.version_manager.worker_versions.clear()

        client.post(
            "/training/version/update",
            json={"worker_id": "worker_0", "version": 10, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_1", "version": 15, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_2", "version": 5, "is_healthy": True}
        )

        # Get stats
        response = client.get("/training/version/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["num_workers"] == 3
        assert data["global_version"] == 10  # Median
        assert data["min_version"] == 5
        assert data["max_version"] == 15
        assert data["mean_version"] == 10.0
        assert data["median_version"] == 10
        assert data["version_range"] == 10

    def test_get_all_workers(self, client):
        """Test GET /training/version/workers endpoint."""
        # Clear and add workers
        server_module.version_manager.worker_versions.clear()

        client.post(
            "/training/version/update",
            json={"worker_id": "worker_0", "version": 10, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_1", "version": 12, "is_healthy": False}
        )

        # Get all workers
        response = client.get("/training/version/workers")
        assert response.status_code == 200
        data = response.json()
        assert len(data["workers"]) == 2

        worker_ids = {w["worker_id"] for w in data["workers"]}
        assert worker_ids == {"worker_0", "worker_1"}

        # Check health status
        worker_1 = next(w for w in data["workers"] if w["worker_id"] == "worker_1")
        assert worker_1["is_healthy"] is False

    def test_get_slow_workers(self, client):
        """Test GET /training/version/slow-workers endpoint."""
        # Clear and add workers
        server_module.version_manager.worker_versions.clear()

        client.post(
            "/training/version/update",
            json={"worker_id": "worker_0", "version": 10, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_1", "version": 15, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_2", "version": 5, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_3", "version": 8, "is_healthy": True}
        )

        # Get slow workers
        response = client.get("/training/version/slow-workers")
        assert response.status_code == 200
        data = response.json()

        # Global version is median of [5, 8, 10, 15] = 9
        # Slow workers are those below 9
        assert len(data["slow_workers"]) == 2
        assert "worker_2" in data["slow_workers"]  # version 5
        assert "worker_3" in data["slow_workers"]  # version 8

    def test_get_ahead_workers(self, client):
        """Test GET /training/version/ahead-workers endpoint."""
        # Clear and add workers
        server_module.version_manager.worker_versions.clear()

        client.post(
            "/training/version/update",
            json={"worker_id": "worker_0", "version": 10, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_1", "version": 12, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "worker_2", "version": 25, "is_healthy": True}
        )

        # Get ahead workers
        response = client.get("/training/version/ahead-workers")
        assert response.status_code == 200
        data = response.json()

        # Global version = median([10, 12, 25]) = 12
        # Staleness bound = 5 (default)
        # Threshold = 12 + 5 = 17
        # worker_2 at 25 > 17, so it's ahead
        assert len(data["ahead_workers"]) == 1
        assert "worker_2" in data["ahead_workers"]

    def test_assign_backups(self, client):
        """Test POST /training/version/assign-backups endpoint."""
        # Clear and add workers
        server_module.version_manager.worker_versions.clear()

        # Create mix of slow and fast workers
        client.post(
            "/training/version/update",
            json={"worker_id": "slow_1", "version": 5, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "slow_2", "version": 7, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "normal", "version": 10, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "fast_1", "version": 20, "is_healthy": True}
        )
        client.post(
            "/training/version/update",
            json={"worker_id": "fast_2", "version": 18, "is_healthy": True}
        )

        # Assign backups
        response = client.post("/training/version/assign-backups")
        assert response.status_code == 200
        data = response.json()

        # Should assign 2 fast workers to 2 slow workers
        assert len(data["assignments"]) == 2

        # Check that fast workers are assigned
        assert "fast_1" in data["assignments"] or "fast_2" in data["assignments"]

        # Check that slow workers are targets
        targets = set(data["assignments"].values())
        assert "slow_1" in targets or "slow_2" in targets

    def test_version_endpoints_integration(self, client):
        """Test full workflow of version management endpoints."""
        # Clear state
        server_module.version_manager.worker_versions.clear()

        # 1. Workers report their versions
        for i in range(4):
            client.post(
                "/training/version/update",
                json={
                    "worker_id": f"worker_{i}",
                    "version": 10 + i * 5,  # 10, 15, 20, 25
                    "is_healthy": True
                }
            )

        # 2. Check global version
        response = client.get("/training/version/global")
        global_version = response.json()["global_version"]
        # Median of [10, 15, 20, 25] = 17.5 -> 17 (int conversion)
        assert global_version == 17

        # 3. Get statistics
        response = client.get("/training/version/stats")
        stats = response.json()
        assert stats["num_workers"] == 4
        assert stats["version_range"] == 15

        # 4. Identify slow workers
        response = client.get("/training/version/slow-workers")
        slow_workers = response.json()["slow_workers"]
        # Workers below global version (17): worker_0 (10), worker_1 (15)
        assert len(slow_workers) == 2

        # 5. Identify ahead workers
        response = client.get("/training/version/ahead-workers")
        ahead_workers = response.json()["ahead_workers"]
        # Threshold = 17 + 5 = 22
        # Workers ahead: worker_3 (25)
        assert len(ahead_workers) == 1
        assert "worker_3" in ahead_workers

        # 6. Assign work stealing
        response = client.post("/training/version/assign-backups")
        assignments = response.json()["assignments"]
        # worker_3 should be assigned to help a slow worker
        assert "worker_3" in assignments
        assert assignments["worker_3"] in ["worker_0", "worker_1"]
