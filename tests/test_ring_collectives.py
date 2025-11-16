"""
Tests for ring-based collective operations.
"""

import pytest
import torch
from communication.ring_collectives import (
    RingCollectiveOps,
    compute_ring_communication_cost
)
from communication.grpc_client import WorkerGRPCClient


class TestRingTopology:
    """Test ring topology setup and neighbor calculations."""

    def test_ring_topology_two_workers(self):
        """Test ring topology with 2 workers."""
        client = WorkerGRPCClient(worker_id="worker_0")

        ring_ops_0 = RingCollectiveOps(
            rank=0,
            world_size=2,
            worker_id="worker_0",
            worker_addresses=["localhost:50051", "localhost:50052"],
            grpc_client=client
        )

        ring_ops_1 = RingCollectiveOps(
            rank=1,
            world_size=2,
            worker_id="worker_1",
            worker_addresses=["localhost:50051", "localhost:50052"],
            grpc_client=client
        )

        # Worker 0: left=1, right=1 (in a 2-worker ring)
        assert ring_ops_0.left_neighbor == 1
        assert ring_ops_0.right_neighbor == 1

        # Worker 1: left=0, right=0
        assert ring_ops_1.left_neighbor == 0
        assert ring_ops_1.right_neighbor == 0

    def test_ring_topology_four_workers(self):
        """Test ring topology with 4 workers."""
        client = WorkerGRPCClient(worker_id="worker_0")
        addresses = [f"localhost:5005{i}" for i in range(4)]

        # Test each worker's neighbors
        expected_neighbors = [
            (3, 1),  # Worker 0: left=3, right=1
            (0, 2),  # Worker 1: left=0, right=2
            (1, 3),  # Worker 2: left=1, right=3
            (2, 0),  # Worker 3: left=2, right=0
        ]

        for rank, (exp_left, exp_right) in enumerate(expected_neighbors):
            ring_ops = RingCollectiveOps(
                rank=rank,
                world_size=4,
                worker_id=f"worker_{rank}",
                worker_addresses=addresses,
                grpc_client=client
            )

            assert ring_ops.left_neighbor == exp_left
            assert ring_ops.right_neighbor == exp_right

    def test_invalid_addresses_count(self):
        """Test that mismatched addresses count raises error."""
        client = WorkerGRPCClient(worker_id="worker_0")

        with pytest.raises(ValueError):
            RingCollectiveOps(
                rank=0,
                world_size=4,
                worker_id="worker_0",
                worker_addresses=["localhost:50051", "localhost:50052"],  # Only 2 addresses
                grpc_client=client
            )


class TestBandwidthAnalysis:
    """Test bandwidth calculation for ring collectives."""

    def test_bandwidth_calculation_4_workers(self):
        """Test bandwidth calculation for 4 workers."""
        world_size = 4
        tensor_size = 1024 * 1024  # 1MB

        cost = compute_ring_communication_cost(world_size, tensor_size)

        # Ring all-reduce: 2(N-1) = 6 steps
        assert cost['ring_steps'] == 6

        # Each step transfers 1/N of the data
        assert cost['ring_data_per_step_bytes'] == tensor_size / world_size

        # Total ring bandwidth: 6 * (1MB / 4) = 1.5 MB
        expected_ring_bandwidth = 6 * (tensor_size / 4)
        assert cost['ring_total_bandwidth_bytes'] == expected_ring_bandwidth

        # Naive all-reduce: N * (N-1) * size = 4 * 3 * 1MB = 12 MB
        expected_naive_bandwidth = world_size * (world_size - 1) * tensor_size
        assert cost['naive_total_bandwidth_bytes'] == expected_naive_bandwidth

        # Bandwidth reduction should be 12 / 1.5 = 8x
        assert cost['bandwidth_reduction_factor'] == pytest.approx(8.0)

    def test_bandwidth_scaling(self):
        """Test that bandwidth reduction improves with more workers."""
        tensor_size = 1024 * 1024

        # Calculate for different world sizes
        results = {}
        for world_size in [2, 4, 8, 16]:
            cost = compute_ring_communication_cost(world_size, tensor_size)
            results[world_size] = cost['bandwidth_reduction_factor']

        # Bandwidth reduction should increase with more workers
        assert results[4] > results[2]
        assert results[8] > results[4]
        assert results[16] > results[8]

    def test_bandwidth_calculation_edge_cases(self):
        """Test edge cases for bandwidth calculation."""
        # Single worker (no communication needed)
        cost_1 = compute_ring_communication_cost(1, 1024)
        assert cost_1['ring_steps'] == 0
        assert cost_1['ring_total_bandwidth_bytes'] == 0

        # Two workers
        cost_2 = compute_ring_communication_cost(2, 1024)
        assert cost_2['ring_steps'] == 2  # 2(N-1) = 2(1) = 2

        # Large world size
        cost_large = compute_ring_communication_cost(100, 1024 * 1024)
        assert cost_large['ring_steps'] == 198  # 2(100-1)


class TestRingCollectiveOpsUnit:
    """Unit tests for ring collective operations (without actual communication)."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test RingCollectiveOps initialization."""
        client = WorkerGRPCClient(worker_id="worker_0")

        ring_ops = RingCollectiveOps(
            rank=0,
            world_size=4,
            worker_id="worker_0",
            worker_addresses=[f"localhost:5005{i}" for i in range(4)],
            grpc_client=client,
            timeout=30.0
        )

        assert ring_ops.rank == 0
        assert ring_ops.world_size == 4
        assert ring_ops.timeout == 30.0
        assert len(ring_ops.worker_addresses) == 4

        await client.close()

    @pytest.mark.asyncio
    async def test_single_worker_passthrough(self):
        """Test that single-worker ring operations are pass-through."""
        client = WorkerGRPCClient(worker_id="worker_0")

        ring_ops = RingCollectiveOps(
            rank=0,
            world_size=1,
            worker_id="worker_0",
            worker_addresses=["localhost:50051"],
            grpc_client=client
        )

        # Test tensor
        tensor = torch.randn(100)

        # All-gather with single worker should return clone
        result_gather = await ring_ops.ring_all_gather_async(tensor)
        assert torch.allclose(result_gather, tensor)

        # All-reduce with single worker should return clone
        result_reduce = await ring_ops.ring_all_reduce_async(tensor)
        assert torch.allclose(result_reduce, tensor)

        # Reduce-scatter with single worker should return clone
        result_scatter = await ring_ops.ring_reduce_scatter_async(tensor)
        assert torch.allclose(result_scatter, tensor)

        await client.close()


class TestRingVsNaiveBandwidth:
    """Compare bandwidth requirements of ring vs naive collectives."""

    def test_bandwidth_comparison_small_cluster(self):
        """Compare bandwidth for small cluster (4 workers, 100MB model)."""
        world_size = 4
        model_size = 100 * 1024 * 1024  # 100 MB

        cost = compute_ring_communication_cost(world_size, model_size)

        # Ring: 2(N-1) * (size/N) = 6 * 25MB = 150MB total
        ring_mb = cost['ring_total_bandwidth_bytes'] / (1024 * 1024)
        assert ring_mb == pytest.approx(150.0)

        # Naive: N*(N-1)*size = 4*3*100MB = 1200MB total
        naive_mb = cost['naive_total_bandwidth_bytes'] / (1024 * 1024)
        assert naive_mb == pytest.approx(1200.0)

        # 8x reduction
        assert cost['bandwidth_reduction_factor'] == pytest.approx(8.0)

    def test_bandwidth_comparison_large_cluster(self):
        """Compare bandwidth for large cluster (32 workers, 1GB model)."""
        world_size = 32
        model_size = 1024 * 1024 * 1024  # 1 GB

        cost = compute_ring_communication_cost(world_size, model_size)

        # Ring: 2(31) * (1GB/32) = 62 * 32MB ≈ 1.94 GB
        ring_gb = cost['ring_total_bandwidth_bytes'] / (1024 ** 3)
        assert ring_gb == pytest.approx(1.9375)

        # Naive: 32*31*1GB = 992 GB
        naive_gb = cost['naive_total_bandwidth_bytes'] / (1024 ** 3)
        assert naive_gb == pytest.approx(992.0)

        # ~512x reduction
        assert cost['bandwidth_reduction_factor'] == pytest.approx(512.0)

    def test_bandwidth_at_different_scales(self):
        """Test bandwidth requirements at different scales."""
        tensor_size = 500 * 1024 * 1024  # 500 MB

        test_cases = [
            (4, 8.0),      # 4 workers: 8x reduction
            (8, 32.0),     # 8 workers: 32x reduction
            (16, 128.0),   # 16 workers: 128x reduction
            (32, 512.0),   # 32 workers: 512x reduction
        ]

        for world_size, expected_reduction in test_cases:
            cost = compute_ring_communication_cost(world_size, tensor_size)
            assert cost['bandwidth_reduction_factor'] == pytest.approx(expected_reduction)


def test_ring_communication_demonstration():
    """
    Demonstrate the bandwidth savings of ring-based collectives.

    This test documents the expected communication patterns.
    """
    # Example: Training a 1B parameter model (4GB) on 16 workers

    params = 1_000_000_000  # 1 billion parameters
    bytes_per_param = 4  # FP32
    model_size = params * bytes_per_param  # 4GB

    world_size = 16

    cost = compute_ring_communication_cost(world_size, model_size)

    # Ring all-reduce communication
    ring_gb = cost['ring_total_bandwidth_bytes'] / (1024 ** 3)

    # Naive all-reduce communication
    naive_gb = cost['naive_total_bandwidth_bytes'] / (1024 ** 3)

    print(f"\n{'=' * 60}")
    print(f"Communication Analysis: 1B param model on {world_size} workers")
    print(f"{'=' * 60}")
    print(f"Model size: {model_size / (1024**3):.2f} GB")
    print(f"Ring all-reduce: {ring_gb:.2f} GB total")
    print(f"Naive all-reduce: {naive_gb:.2f} GB total")
    print(f"Bandwidth savings: {cost['bandwidth_reduction_factor']:.1f}x")
    print(f"{'=' * 60}\n")

    # Verify calculations
    assert ring_gb < naive_gb
    assert cost['bandwidth_reduction_factor'] > 1
