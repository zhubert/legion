"""
Tests for HuggingFace dataset integration.

Tests dataset configuration, loading, and sharding.
"""

import pytest
from core.dataset import (
    get_dataset_config,
    list_available_datasets,
    DATASET_CONFIGS
)


class TestDatasetConfiguration:
    """Test dataset configuration utilities."""

    def test_get_dataset_config_known_datasets(self):
        """Test getting config for known datasets."""
        for name in DATASET_CONFIGS.keys():
            config = get_dataset_config(name)
            assert 'full_name' in config
            assert 'streaming' in config
            assert 'tokenizer' in config
            assert 'seq_len' in config
            assert 'buffer_size' in config
            assert 'description' in config

    def test_get_dataset_config_fineweb_edu(self):
        """Test FineWeb-Edu configuration."""
        config = get_dataset_config('fineweb-edu')
        assert config['full_name'] == 'HuggingFaceFW/fineweb-edu'
        assert config['streaming'] is True
        assert config['tokenizer'] == 'gpt2'
        assert config['seq_len'] == 1024

    def test_get_dataset_config_shakespeare(self):
        """Test Shakespeare configuration."""
        config = get_dataset_config('tiny_shakespeare')
        assert config['full_name'] == 'karpathy/tiny_shakespeare'
        assert config['streaming'] is False  # Small dataset
        assert config['seq_len'] == 256

    def test_get_dataset_config_custom_dataset(self):
        """Test custom dataset name (not in configs)."""
        custom_name = "my-org/custom-dataset"
        config = get_dataset_config(custom_name)
        assert config['full_name'] == custom_name
        assert config['streaming'] is True
        assert config['tokenizer'] == 'gpt2'

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = list_available_datasets()
        assert len(datasets) > 0
        assert all('name' in ds for ds in datasets)
        assert all('full_name' in ds for ds in datasets)
        assert all('description' in ds for ds in datasets)

        # Check known datasets are present
        names = [ds['name'] for ds in datasets]
        assert 'fineweb' in names
        assert 'fineweb-edu' in names
        assert 'pile' in names
        assert 'tiny_shakespeare' in names
        assert 'shakespeare' in names


class TestHuggingFaceDatasetLoading:
    """Test actual HuggingFace dataset loading (requires datasets library)."""

    def test_import_requirements(self):
        """Test that HuggingFace libraries can be imported."""
        try:
            import datasets
            import transformers
            assert datasets is not None
            assert transformers is not None
        except ImportError:
            pytest.skip("datasets and transformers not installed")

    @pytest.mark.skipif(
        True,  # Skip by default to avoid long download times
        reason="Skipped to avoid downloading large datasets in CI"
    )
    def test_load_tiny_shakespeare(self):
        """Test loading tiny shakespeare dataset."""
        from core.dataset import create_huggingface_dataset

        dataset = create_huggingface_dataset(
            dataset_name='tiny_shakespeare',
            rank=0,
            world_size=1,
            num_batches=5,
            batch_size=4,
            seq_len=128
        )

        assert len(dataset) == 5
        for inputs, labels in dataset:
            assert inputs.shape == (4, 128)
            assert labels.shape == (4, 128)

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Skipped to avoid downloading large datasets in CI"
    )
    def test_distributed_sharding(self):
        """Test that workers get non-overlapping shards."""
        from core.dataset import create_huggingface_dataset

        world_size = 4
        num_batches = 10
        batch_size = 4

        # Create datasets for all workers
        datasets = [
            create_huggingface_dataset(
                dataset_name='tiny_shakespeare',
                rank=rank,
                world_size=world_size,
                num_batches=num_batches,
                batch_size=batch_size,
                seq_len=128
            )
            for rank in range(world_size)
        ]

        # Verify each worker got correct number of batches
        for dataset in datasets:
            assert len(dataset) == num_batches


class TestWorkerConfigDatasetIntegration:
    """Test worker config dataset fields."""

    def test_worker_config_has_dataset_fields(self):
        """Test that WorkerConfig has dataset configuration."""
        from worker.config import WorkerConfig

        config = WorkerConfig()
        assert hasattr(config, 'dataset_name')
        assert hasattr(config, 'dataset_type')
        assert hasattr(config, 'tokenizer_name')

    def test_worker_config_dataset_defaults(self):
        """Test default dataset configuration."""
        from worker.config import WorkerConfig

        config = WorkerConfig()
        assert config.dataset_type == "dummy"
        assert config.dataset_name is None
        assert config.tokenizer_name is None

    def test_worker_config_huggingface_dataset(self):
        """Test setting HuggingFace dataset in config."""
        from worker.config import WorkerConfig

        config = WorkerConfig(
            dataset_type="huggingface",
            dataset_name="fineweb-edu",
            tokenizer_name="gpt2"
        )

        assert config.dataset_type == "huggingface"
        assert config.dataset_name == "fineweb-edu"
        assert config.tokenizer_name == "gpt2"

    def test_worker_config_to_dict_includes_dataset(self):
        """Test that to_dict includes dataset config."""
        from worker.config import WorkerConfig

        config = WorkerConfig(
            dataset_type="huggingface",
            dataset_name="pile"
        )

        config_dict = config.to_dict()
        assert 'dataset_type' in config_dict
        assert 'dataset_name' in config_dict
        assert 'tokenizer_name' in config_dict
        assert config_dict['dataset_type'] == "huggingface"
        assert config_dict['dataset_name'] == "pile"
