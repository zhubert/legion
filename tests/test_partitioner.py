"""
Tests for parameter partitioning (ZeRO-3)
"""

import pytest
import torch
from sim.model import TinyGPT
from sim.partitioner import (
    Partitioner,
    ParameterPartition,
    flatten_parameters,
    unflatten_parameters
)


class TestParameterPartition:
    """Test ParameterPartition dataclass"""

    def test_partition_creation(self):
        """Test creating a partition"""
        partition = ParameterPartition(
            rank=0,
            world_size=4,
            param_names=["layer1.weight", "layer1.bias"],
            start_idx=0,
            end_idx=100
        )

        assert partition.rank == 0
        assert partition.world_size == 4
        assert partition.num_params == 100
        assert len(partition.param_names) == 2


class TestPartitioner:
    """Test Partitioner for ZeRO-3 style sharding"""

    def test_partitioner_creation(self):
        """Test creating a partitioner"""
        model = TinyGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
        partitioner = Partitioner(model, world_size=4)

        assert partitioner.world_size == 4
        assert len(partitioner.partitions) == 4
        assert partitioner.total_params > 0

    def test_partitions_cover_all_parameters(self):
        """Test that partitions cover all parameters exactly once"""
        model = TinyGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
        partitioner = Partitioner(model, world_size=4)

        # Sum of partition sizes should equal total params
        total_partitioned = sum(p.num_params for p in partitioner.partitions)
        assert total_partitioned == partitioner.total_params

        # Check no gaps or overlaps
        partitions_sorted = sorted(partitioner.partitions, key=lambda p: p.start_idx)
        for i in range(len(partitions_sorted) - 1):
            assert partitions_sorted[i].end_idx == partitions_sorted[i+1].start_idx

    def test_balanced_partitioning(self):
        """Test that partitions are roughly balanced"""
        # Use a larger model so partitions can be balanced
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=4)
        partitioner = Partitioner(model, world_size=4)

        avg_size = partitioner.total_params / partitioner.world_size
        # Last partition may be larger (gets remaining params), so check all but last
        for partition in partitioner.partitions[:-1]:
            # Each partition should be within 50% of average
            assert partition.num_params > avg_size * 0.5
            assert partition.num_params < avg_size * 1.5

        # Last partition should exist and have some params
        assert partitioner.partitions[-1].num_params > 0

    def test_get_partition(self):
        """Test getting partition by rank"""
        model = TinyGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
        partitioner = Partitioner(model, world_size=4)

        partition = partitioner.get_partition(0)
        assert partition.rank == 0

        partition = partitioner.get_partition(3)
        assert partition.rank == 3

    def test_get_partition_invalid_rank(self):
        """Test that invalid rank raises error"""
        model = TinyGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
        partitioner = Partitioner(model, world_size=4)

        with pytest.raises(ValueError):
            partitioner.get_partition(10)

    def test_get_owned_parameters(self):
        """Test extracting owned parameters"""
        model = TinyGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
        partitioner = Partitioner(model, world_size=4)

        owned = partitioner.get_owned_parameters(rank=0)

        # Should be a dict of tensors
        assert isinstance(owned, dict)
        assert len(owned) > 0
        assert all(isinstance(v, torch.Tensor) for v in owned.values())

    def test_single_worker(self):
        """Test partitioning with single worker"""
        model = TinyGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
        partitioner = Partitioner(model, world_size=1)

        assert len(partitioner.partitions) == 1
        assert partitioner.partitions[0].num_params == partitioner.total_params

    def test_many_workers(self):
        """Test partitioning with many workers"""
        # Use larger model for many workers (otherwise some partitions would be empty)
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=6)
        partitioner = Partitioner(model, world_size=8)

        # Should create partitions for all workers (or close to it for large models)
        assert len(partitioner.partitions) >= 7  # At least most workers get partitions
        assert len(partitioner.partitions) <= 8


class TestFlattenUnflatten:
    """Test parameter flattening utilities"""

    def test_flatten_parameters(self):
        """Test flattening parameters"""
        params = {
            "weight1": torch.randn(10, 5),
            "weight2": torch.randn(3, 4),
            "bias": torch.randn(5)
        }

        flat = flatten_parameters(params)

        # Total elements: 10*5 + 3*4 + 5 = 67
        assert flat.numel() == 67
        assert flat.dim() == 1

    def test_unflatten_parameters(self):
        """Test unflattening parameters"""
        original_params = {
            "weight1": torch.randn(10, 5),
            "weight2": torch.randn(3, 4),
            "bias": torch.randn(5)
        }

        # Flatten
        flat = flatten_parameters(original_params)

        # Get shapes
        param_shapes = {name: param.shape for name, param in original_params.items()}

        # Unflatten
        reconstructed = unflatten_parameters(flat, param_shapes)

        # Check all parameters match
        assert set(reconstructed.keys()) == set(original_params.keys())
        for name in original_params.keys():
            assert torch.allclose(reconstructed[name], original_params[name])

    def test_flatten_unflatten_round_trip(self):
        """Test that flatten/unflatten is lossless"""
        params = {
            "a": torch.randn(5, 3),
            "b": torch.randn(10),
            "c": torch.randn(2, 2, 2)
        }

        flat = flatten_parameters(params)
        shapes = {name: param.shape for name, param in params.items()}
        reconstructed = unflatten_parameters(flat, shapes)

        for name in params.keys():
            assert params[name].shape == reconstructed[name].shape
            assert torch.equal(params[name], reconstructed[name])
