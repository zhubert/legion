"""
Integration tests for work stealing in async parameter server.

Tests the complete flow:
1. Fast worker gets ahead of staleness bound
2. Coordinator assigns it to help slow worker
3. Fast worker computes backup gradients
4. Slow worker receives extra gradient contributions
"""

import pytest
import asyncio
import torch
import httpx

from coordinator.server import app
from coordinator.version_manager import VersionManager
import coordinator.server as server_module

from communication.grpc_server import WorkerGRPCServer
from communication.grpc_client import WorkerGRPCClient
from communication.async_collectives import AsyncParameterFetcher, AsyncGradientPusher


@pytest.mark.asyncio
async def test_work_stealing_assignment():
    """Test that coordinator assigns fast workers to help slow workers."""
    # Initialize version manager
    vm = VersionManager(staleness_bound=5)

    # Simulate worker versions: worker_0 (fast), worker_1 (slow)
    vm.update_worker_version("worker_0", version=20, is_healthy=True)  # Ahead
    vm.update_worker_version("worker_1", version=8, is_healthy=True)   # Behind

    # Global version = median([20, 8]) = 14
    # Staleness bound = 5
    # Threshold = 14 + 5 = 19
    # worker_0 at 20 > 19, so it's ahead

    global_version = vm.get_global_version()
    assert global_version == 14

    # Check that worker_0 is ahead
    is_ahead = vm.is_worker_too_far_ahead("worker_0")
    assert is_ahead is True

    # Check that worker_1 is slow
    slow_workers = vm.get_slow_workers(global_version)
    assert "worker_1" in slow_workers

    # Assign work stealing
    assignments = vm.assign_work_stealing()

    # worker_0 should be assigned to help worker_1
    assert "worker_0" in assignments
    assert assignments["worker_0"] == "worker_1"


@pytest.mark.asyncio
async def test_backup_gradient_computation():
    """Test that fast worker can compute backup gradients for slow worker."""
    # Create two parameter servers (simulating two workers)
    param_store_0 = {"layer.weight": torch.randn(10, 5)}
    param_store_1 = {"layer.bias": torch.randn(5)}

    server_0 = WorkerGRPCServer(
        worker_id="worker_0",
        parameter_store=param_store_0,
        host="127.0.0.1",
        port=50051
    )
    server_1 = WorkerGRPCServer(
        worker_id="worker_1",
        parameter_store=param_store_1,
        host="127.0.0.1",
        port=50052
    )

    # Configure for gradient accumulation
    server_0.servicer.set_world_size(2)
    server_1.servicer.set_world_size(2)
    server_0.servicer.set_aggregation_threshold(0.5)  # 50% = 1/2 workers
    server_1.servicer.set_aggregation_threshold(0.5)

    try:
        await server_0.start()
        await server_1.start()
        await asyncio.sleep(0.2)

        # Create client (fast worker)
        client = WorkerGRPCClient(worker_id="worker_fast")
        fetcher = AsyncParameterFetcher(client, timeout=5.0)
        pusher = AsyncGradientPusher(client, timeout=5.0)

        # === Simulate backup gradient computation ===

        # 1. Fetch parameters for slow worker's version (v8)
        parameter_requests = [
            ("127.0.0.1:50051", "layer.weight", 0, -1),
            ("127.0.0.1:50052", "layer.bias", 0, -1),
        ]

        fetched_params = await fetcher.fetch_parameters_async(
            parameter_requests,
            version=8,
            staleness_tolerance=5
        )

        assert "layer.weight[0:-1]" in fetched_params
        assert "layer.bias[0:-1]" in fetched_params
        assert fetched_params["layer.weight[0:-1]"] is not None
        assert fetched_params["layer.bias[0:-1]"] is not None

        # 2. Compute fake gradients (in real case, would run forward/backward)
        grad_weight = torch.randn(10, 5)
        grad_bias = torch.randn(5)

        # 3. Push backup gradients tagged with slow worker's version
        gradient_requests = [
            ("127.0.0.1:50051", "layer.weight", grad_weight, 0, -1),
            ("127.0.0.1:50052", "layer.bias", grad_bias, 0, -1),
        ]

        results = await pusher.push_gradients_async(
            gradient_requests,
            version=8  # Slow worker's version
        )

        # Verify gradients were pushed successfully
        assert results["layer.weight[0:-1]"] is True
        assert results["layer.bias[0:-1]"] is True

        # 4. Verify gradients were accumulated on servers for version 8
        # Check server_0 received gradient for layer.weight at v8
        accumulated_weight = await server_0.servicer.get_accumulated_gradients(
            version=8,
            param_name="layer.weight",
            wait_for_threshold=False,
            timeout=1.0
        )

        assert accumulated_weight is not None
        assert accumulated_weight.shape == (10, 5)

        # Check server_1 received gradient for layer.bias at v8
        accumulated_bias = await server_1.servicer.get_accumulated_gradients(
            version=8,
            param_name="layer.bias",
            wait_for_threshold=False,
            timeout=1.0
        )

        assert accumulated_bias is not None
        assert accumulated_bias.shape == (5,)

    finally:
        await server_0.stop()
        await server_1.stop()
        await client.close()


@pytest.mark.asyncio
async def test_multiple_backup_contributions():
    """Test that slow worker receives multiple backup gradient contributions."""
    # Create parameter server for slow worker
    param_store = {"model.weight": torch.randn(10, 10)}

    server = WorkerGRPCServer(
        worker_id="worker_slow",
        parameter_store=param_store,
        host="127.0.0.1",
        port=50051
    )

    # Configure to expect 3 workers (1 slow + 2 fast doing backup)
    server.servicer.set_world_size(3)
    server.servicer.set_aggregation_threshold(0.67)  # 67% = 2/3 workers

    try:
        await server.start()
        await asyncio.sleep(0.2)

        # Slow worker sends its own gradient
        slow_client = WorkerGRPCClient(worker_id="worker_slow")
        slow_pusher = AsyncGradientPusher(slow_client)

        slow_grad = torch.ones(10, 10) * 1.0
        results = await slow_pusher.push_gradients_async(
            [("127.0.0.1:50051", "model.weight", slow_grad, 0, -1)],
            version=8
        )
        assert results["model.weight[0:-1]"] is True

        # Fast worker 1 sends backup gradient
        fast_client_1 = WorkerGRPCClient(worker_id="worker_fast_1")
        fast_pusher_1 = AsyncGradientPusher(fast_client_1)

        backup_grad_1 = torch.ones(10, 10) * 2.0
        results = await fast_pusher_1.push_gradients_async(
            [("127.0.0.1:50051", "model.weight", backup_grad_1, 0, -1)],
            version=8  # Same version as slow worker
        )
        assert results["model.weight[0:-1]"] is True

        # Now 2/3 workers have contributed, should meet threshold
        accumulated = await server.servicer.get_accumulated_gradients(
            version=8,
            param_name="model.weight",
            wait_for_threshold=True,
            timeout=2.0
        )

        assert accumulated is not None
        # Should be sum of slow worker (1.0) + fast worker 1 (2.0) = 3.0 per element
        expected = torch.ones(10, 10) * 3.0
        assert torch.allclose(accumulated, expected)

        # Fast worker 2 sends another backup gradient (after threshold already met)
        fast_client_2 = WorkerGRPCClient(worker_id="worker_fast_2")
        fast_pusher_2 = AsyncGradientPusher(fast_client_2)

        backup_grad_2 = torch.ones(10, 10) * 3.0
        results = await fast_pusher_2.push_gradients_async(
            [("127.0.0.1:50051", "model.weight", backup_grad_2, 0, -1)],
            version=8
        )
        assert results["model.weight[0:-1]"] is True

        # Re-fetch accumulated gradients (should now include all 3)
        accumulated_all = await server.servicer.get_accumulated_gradients(
            version=8,
            param_name="model.weight",
            wait_for_threshold=False,
            timeout=1.0
        )

        # Should be 1.0 + 2.0 + 3.0 = 6.0 per element
        expected_all = torch.ones(10, 10) * 6.0
        assert torch.allclose(accumulated_all, expected_all)

    finally:
        await server.stop()
        await slow_client.close()
        await fast_client_1.close()
        await fast_client_2.close()


@pytest.mark.asyncio
async def test_work_stealing_with_coordinator():
    """Test full work stealing flow with coordinator integration."""
    # This would be a full end-to-end test with actual worker processes
    # For now, test the coordinator API for work stealing

    vm = VersionManager(staleness_bound=5)

    # Setup: 1 fast worker, 2 slow workers
    vm.update_worker_version("worker_fast", version=25, is_healthy=True)
    vm.update_worker_version("worker_slow_1", version=8, is_healthy=True)
    vm.update_worker_version("worker_slow_2", version=10, is_healthy=True)

    # Global version = median([25, 8, 10]) = 10
    global_version = vm.get_global_version()
    assert global_version == 10

    # Fast worker should be ahead (25 > 10 + 5)
    assert vm.is_worker_too_far_ahead("worker_fast") is True

    # Assign work stealing
    assignments = vm.assign_work_stealing()

    # Fast worker should be assigned to slowest worker
    assert "worker_fast" in assignments
    assert assignments["worker_fast"] == "worker_slow_1"  # Version 8 is slowest

    # Verify fast worker gets assignment through API
    backup_assignment = vm.get_backup_assignment("worker_fast")
    assert backup_assignment == "worker_slow_1"

    # Verify slow workers are not assigned (they're not ahead)
    assert vm.get_backup_assignment("worker_slow_1") is None
    assert vm.get_backup_assignment("worker_slow_2") is None


@pytest.mark.asyncio
async def test_work_stealing_prevents_further_ahead():
    """Test that work stealing prevents fast worker from getting further ahead."""
    vm = VersionManager(staleness_bound=5)

    # Initial state: worker_0 at version 20, worker_1 at version 8
    vm.update_worker_version("worker_0", version=20, is_healthy=True)
    vm.update_worker_version("worker_1", version=8, is_healthy=True)

    # worker_0 is ahead
    assert vm.is_worker_too_far_ahead("worker_0") is True

    # Assign work stealing
    assignments = vm.assign_work_stealing()
    assert assignments["worker_0"] == "worker_1"

    # In real training loop, worker_0 would:
    # 1. Compute backup gradients for worker_1's version (8)
    # 2. Skip its own training step (stays at version 20)
    # 3. Wait until global version catches up

    # Simulate slow worker catching up
    vm.update_worker_version("worker_1", version=12, is_healthy=True)

    # Global version now = median([20, 12]) = 16
    # worker_0 at 20 is still ahead (20 > 16 + 5 = 21... wait, 20 < 21)
    # Actually worker_0 is no longer ahead!

    global_version = vm.get_global_version()
    assert global_version == 16

    # worker_0 should no longer be ahead (20 <= 16 + 5 = 21)
    assert vm.is_worker_too_far_ahead("worker_0") is False

    # worker_0 can now continue its own training


@pytest.mark.asyncio
async def test_no_work_stealing_when_homogeneous():
    """Test that work stealing is not triggered when all workers are similar speed."""
    vm = VersionManager(staleness_bound=5)

    # All workers at similar versions
    vm.update_worker_version("worker_0", version=100, is_healthy=True)
    vm.update_worker_version("worker_1", version=101, is_healthy=True)
    vm.update_worker_version("worker_2", version=99, is_healthy=True)
    vm.update_worker_version("worker_3", version=100, is_healthy=True)

    # Global version = median([99, 100, 100, 101]) = 100
    global_version = vm.get_global_version()
    assert global_version == 100

    # No worker should be ahead (all within staleness bound)
    assert vm.is_worker_too_far_ahead("worker_0") is False
    assert vm.is_worker_too_far_ahead("worker_1") is False  # 101 <= 100 + 5
    assert vm.is_worker_too_far_ahead("worker_2") is False
    assert vm.is_worker_too_far_ahead("worker_3") is False

    # No slow workers (all at or above median)
    slow_workers = vm.get_slow_workers(global_version)
    # worker_2 at 99 is below median 100, so it's slow
    assert len(slow_workers) == 1
    assert "worker_2" in slow_workers

    # But no ahead workers to assign
    ahead_workers = vm.get_ahead_workers(global_version)
    assert len(ahead_workers) == 0

    # No work stealing assignments
    assignments = vm.assign_work_stealing()
    assert len(assignments) == 0
