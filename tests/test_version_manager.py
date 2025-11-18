"""
Tests for version management system.

Tests bounded staleness tracking, work stealing assignment,
and cluster-wide version coordination.
"""

import pytest
import time

from coordinator.version_manager import VersionManager, WorkerVersion


class TestVersionTracking:
    """Test basic version tracking functionality."""

    def test_update_worker_version(self):
        """Test updating worker versions."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=12)
        vm.update_worker_version("worker_2", version=8)

        # Verify versions stored
        assert vm.get_worker_info("worker_0").version == 10
        assert vm.get_worker_info("worker_1").version == 12
        assert vm.get_worker_info("worker_2").version == 8

    def test_global_version_median(self):
        """Test global version computation (median of active workers)."""
        vm = VersionManager(staleness_bound=5)

        # Test with 3 workers
        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=12)
        vm.update_worker_version("worker_2", version=8)

        global_version = vm.get_global_version()
        assert global_version == 10  # Median of [8, 10, 12]

    def test_global_version_median_even_workers(self):
        """Test global version with even number of workers."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=12)
        vm.update_worker_version("worker_2", version=8)
        vm.update_worker_version("worker_3", version=14)

        global_version = vm.get_global_version()
        # Median of [8, 10, 12, 14] = 11 (average of middle two)
        assert global_version == 11

    def test_mark_worker_offline(self):
        """Test removing workers from version tracking."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=12)

        # Mark worker offline
        vm.mark_worker_offline("worker_0")

        # Should only have worker_1
        assert vm.get_worker_info("worker_0") is None
        assert vm.get_worker_info("worker_1") is not None
        assert vm.get_global_version() == 12  # Only worker_1 remains


class TestStalenessDetection:
    """Test staleness detection and bounded staleness enforcement."""

    def test_get_slow_workers(self):
        """Test identification of slow workers (below median)."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)  # At median
        vm.update_worker_version("worker_1", version=15)  # Ahead
        vm.update_worker_version("worker_2", version=5)   # Behind
        vm.update_worker_version("worker_3", version=8)   # Behind

        slow_workers = vm.get_slow_workers()

        # Should return workers below median (10)
        assert len(slow_workers) == 2
        assert "worker_2" in slow_workers  # Version 5
        assert "worker_3" in slow_workers  # Version 8

        # Should be sorted slowest first
        assert slow_workers[0] == "worker_2"  # 5 < 8

    def test_get_ahead_workers(self):
        """Test identification of workers beyond staleness bound."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)  # At median
        vm.update_worker_version("worker_1", version=12)  # Slightly ahead (ok)
        vm.update_worker_version("worker_2", version=20)  # Too far ahead
        vm.update_worker_version("worker_3", version=18)  # Too far ahead

        ahead_workers = vm.get_ahead_workers()

        # Global version (median) = 15 (median of [10, 12, 18, 20] = avg of 12 and 18)
        # Staleness bound = 5
        # Threshold = 15 + 5 = 20
        # Workers beyond 20: worker_2 (20 is NOT beyond, it's AT threshold)
        # Actually none are strictly BEYOND 20

        # Adjust test: use clearer version numbers
        vm2 = VersionManager(staleness_bound=5)
        vm2.update_worker_version("worker_0", version=10)
        vm2.update_worker_version("worker_1", version=12)
        vm2.update_worker_version("worker_2", version=25)  # Clearly too far ahead
        vm2.update_worker_version("worker_3", version=22)  # Clearly too far ahead

        # Median = 17, threshold = 22, workers beyond: worker_2 (25)
        ahead_workers2 = vm2.get_ahead_workers()
        assert len(ahead_workers2) == 1
        assert "worker_2" in ahead_workers2

    def test_is_worker_too_far_ahead(self):
        """Test checking if specific worker is beyond staleness bound."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=12)
        vm.update_worker_version("worker_2", version=20)

        # Global version (median) = 12
        # Threshold = 12 + 5 = 17

        assert vm.is_worker_too_far_ahead("worker_0") is False  # 10 < 17
        assert vm.is_worker_too_far_ahead("worker_1") is False  # 12 < 17
        assert vm.is_worker_too_far_ahead("worker_2") is True   # 20 > 17

    def test_is_worker_too_far_behind(self):
        """Test checking if worker is lagging too far."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=12)
        vm.update_worker_version("worker_2", version=3)

        # Global version (median) = 10
        # Lag threshold = 10 - 5 = 5

        assert vm.is_worker_too_far_behind("worker_0") is False  # 10 >= 5
        assert vm.is_worker_too_far_behind("worker_1") is False  # 12 >= 5
        assert vm.is_worker_too_far_behind("worker_2") is True   # 3 < 5


class TestWorkStealingAssignment:
    """Test work stealing assignment logic."""

    def test_assign_work_stealing(self):
        """Test assigning ahead workers to help slow workers."""
        vm = VersionManager(staleness_bound=5)

        # Create cluster with mix of speeds
        vm.update_worker_version("slow_1", version=5)   # Slow
        vm.update_worker_version("slow_2", version=7)   # Slow
        vm.update_worker_version("normal", version=10)  # At median
        vm.update_worker_version("fast_1", version=20)  # Ahead
        vm.update_worker_version("fast_2", version=18)  # Ahead

        # Assign work stealing
        assignments = vm.assign_work_stealing()

        # Should assign 2 fast workers to 2 slow workers
        assert len(assignments) == 2
        assert "fast_1" in assignments or "fast_2" in assignments
        assert "slow_1" in assignments.values() or "slow_2" in assignments.values()

    def test_get_backup_assignment(self):
        """Test retrieving backup assignment for specific worker."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("slow", version=5)
        vm.update_worker_version("normal", version=10)
        vm.update_worker_version("fast", version=20)

        assignments = vm.assign_work_stealing()

        # Fast worker should be assigned to help slow worker
        assignment = vm.get_backup_assignment("fast")
        assert assignment == "slow"

        # Normal worker should have no assignment
        assignment = vm.get_backup_assignment("normal")
        assert assignment is None

    def test_work_stealing_cleanup_on_worker_offline(self):
        """Test that backup assignments are cleaned up when workers go offline."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("slow", version=5)
        vm.update_worker_version("fast", version=20)

        assignments = vm.assign_work_stealing()
        assert "fast" in assignments

        # Mark slow worker offline
        vm.mark_worker_offline("slow")

        # Assignment should be cleaned up
        assignment = vm.get_backup_assignment("fast")
        assert assignment is None


class TestVersionStatistics:
    """Test version statistics and monitoring."""

    def test_staleness_stats(self):
        """Test staleness statistics computation."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=15)
        vm.update_worker_version("worker_2", version=5)

        stats = vm.get_staleness_stats()

        assert stats["num_workers"] == 3
        assert stats["global_version"] == 10  # Median
        assert stats["min_version"] == 5
        assert stats["max_version"] == 15
        assert stats["mean_version"] == 10.0  # (5 + 10 + 15) / 3
        assert stats["median_version"] == 10
        assert stats["version_range"] == 10  # 15 - 5
        assert stats["staleness_bound"] == 5

    def test_version_progress_rate(self):
        """Test cluster progress rate computation."""
        vm = VersionManager(staleness_bound=5)

        # Simulate progress over time
        vm.update_worker_version("worker_0", version=0)
        vm.get_global_version()  # Record initial

        time.sleep(0.1)

        vm.update_worker_version("worker_0", version=10)
        vm.get_global_version()  # Record after progress

        rate = vm.get_version_progress_rate(window_seconds=1.0)

        # Should be roughly 10 versions / 0.1 seconds = 100 versions/sec
        # (Allow some tolerance for timing variations)
        assert rate > 50  # At least 50 versions/sec
        assert rate < 200  # Less than 200 versions/sec

    def test_get_all_workers(self):
        """Test retrieving all worker version information."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10, is_healthy=True)
        vm.update_worker_version("worker_1", version=12, is_healthy=True)
        vm.update_worker_version("worker_2", version=8, is_healthy=False)

        workers = vm.get_all_workers()

        assert len(workers) == 3
        worker_ids = {w.worker_id for w in workers}
        assert worker_ids == {"worker_0", "worker_1", "worker_2"}

        # Check health status
        worker_2 = next(w for w in workers if w.worker_id == "worker_2")
        assert worker_2.is_healthy is False


class TestDynamicStalenessBound:
    """Test dynamic staleness bound adjustment."""

    def test_set_staleness_bound(self):
        """Test updating staleness bound."""
        vm = VersionManager(staleness_bound=5)

        assert vm.staleness_bound == 5

        vm.set_staleness_bound(10)
        assert vm.staleness_bound == 10

    def test_staleness_bound_validation(self):
        """Test staleness bound validation."""
        vm = VersionManager(staleness_bound=5)

        with pytest.raises(ValueError):
            vm.set_staleness_bound(0)  # Must be >= 1

        with pytest.raises(ValueError):
            vm.set_staleness_bound(-5)  # Must be >= 1

    def test_staleness_bound_affects_ahead_workers(self):
        """Test that changing staleness bound affects ahead worker detection."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("normal", version=10)
        vm.update_worker_version("ahead", version=20)

        # With staleness_bound=5, global=15 (median of [10, 20]), threshold=20
        # Worker at 20 is NOT ahead (exactly at threshold)
        # Let's use version 25 to be clearly ahead
        vm.update_worker_version("ahead", version=25)

        assert vm.is_worker_too_far_ahead("ahead") is True  # 25 > 20

        # Increase staleness bound to 15
        vm.set_staleness_bound(15)

        # Now threshold=30, worker at 25 is not ahead
        assert vm.is_worker_too_far_ahead("ahead") is False  # 25 < 30


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_cluster(self):
        """Test behavior with no workers."""
        vm = VersionManager(staleness_bound=5)

        assert vm.get_global_version() == 0
        assert vm.get_slow_workers() == []
        assert vm.get_ahead_workers() == []
        assert vm.assign_work_stealing() == {}

        stats = vm.get_staleness_stats()
        assert stats["num_workers"] == 0

    def test_single_worker(self):
        """Test behavior with single worker."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)

        assert vm.get_global_version() == 10
        assert vm.get_slow_workers() == []  # No one below median
        assert vm.get_ahead_workers() == []  # No one ahead
        assert vm.is_worker_too_far_ahead("worker_0") is False

    def test_all_workers_same_version(self):
        """Test behavior when all workers at same version."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("worker_0", version=10)
        vm.update_worker_version("worker_1", version=10)
        vm.update_worker_version("worker_2", version=10)

        assert vm.get_global_version() == 10
        assert vm.get_slow_workers() == []
        assert vm.get_ahead_workers() == []
        assert vm.assign_work_stealing() == {}

    def test_unhealthy_workers_excluded_from_global_version(self):
        """Test that unhealthy workers don't affect global version."""
        vm = VersionManager(staleness_bound=5)

        vm.update_worker_version("healthy_1", version=10, is_healthy=True)
        vm.update_worker_version("healthy_2", version=12, is_healthy=True)
        vm.update_worker_version("unhealthy", version=100, is_healthy=False)

        # Global version should only consider healthy workers
        global_version = vm.get_global_version()
        assert global_version == 11  # Median of [10, 12], ignoring 100
