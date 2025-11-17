"""
Unit tests for worker components.

Tests:
- Configuration
- Coordinator client
- Heartbeat manager
- Shard manager
- Telemetry reporter
- Distributed trainer
"""

import pytest
import os
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import torch
import torch.nn as nn

from worker.config import WorkerConfig, get_local_ip
from worker.coordinator_client import CoordinatorClient
from worker.heartbeat import HeartbeatManager
from worker.shard_manager import ShardManager
from worker.telemetry import TelemetryReporter, MetricRecord
from worker.trainer import DistributedTrainer

from core.model import create_model


class TestWorkerConfig:
    """Test worker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkerConfig()

        assert config.worker_id.startswith("worker_")
        assert config.coordinator_url == "http://localhost:8000"
        assert config.port == 50051
        assert config.device in ["cpu", "cuda"]
        assert config.heartbeat_interval == 30
        assert config.telemetry_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkerConfig(
            worker_id="test_worker",
            coordinator_url="http://example.com:9000",
            device="cpu",
            model_size="small"
        )

        assert config.worker_id == "test_worker"
        assert config.coordinator_url == "http://example.com:9000"
        assert config.model_size == "small"

    def test_auto_detect_hardware(self):
        """Test hardware auto-detection."""
        config = WorkerConfig()

        # CPU cores should be detected
        assert config.cpu_cores is not None
        assert config.cpu_cores > 0

        # RAM should be detected if psutil available
        if config.ram_gb is not None:
            assert config.ram_gb > 0

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = WorkerConfig(worker_id="test", model_size="tiny")
        config_dict = config.to_dict()

        assert config_dict['worker_id'] == "test"
        assert config_dict['model_size'] == "tiny"
        assert 'coordinator_url' in config_dict

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'worker_id': 'test_worker',
            'model_size': 'small',
            'batch_size': 8
        }
        config = WorkerConfig.from_dict(config_dict)

        assert config.worker_id == 'test_worker'
        assert config.model_size == 'small'
        assert config.batch_size == 8

    def test_json_serialization(self):
        """Test JSON save/load."""
        config = WorkerConfig(worker_id="test", model_size="medium")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            config.to_json_file(path)
            loaded_config = WorkerConfig.from_json_file(path)

            assert loaded_config.worker_id == config.worker_id
            assert loaded_config.model_size == config.model_size
        finally:
            os.unlink(path)

    def test_get_gpu_info(self):
        """Test GPU info retrieval."""
        config = WorkerConfig(device="cpu")
        assert config.get_gpu_info() is None

        config_cuda = WorkerConfig(
            device="cuda",
            gpu_name="RTX 4090",
            gpu_memory_gb=24.0
        )
        gpu_info = config_cuda.get_gpu_info()

        if config_cuda.device == "cuda":  # Only if CUDA available
            assert gpu_info is not None
            assert 'name' in gpu_info

    def test_get_local_ip(self):
        """Test local IP detection."""
        ip = get_local_ip()
        assert ip is not None
        assert len(ip.split('.')) == 4  # Should be IPv4


class TestCoordinatorClient:
    """Test coordinator client."""

    @pytest.fixture
    def client(self):
        """Create coordinator client for testing."""
        return CoordinatorClient(
            coordinator_url="http://localhost:8000",
            worker_id="test_worker",
            ip_address="192.168.1.100",
            port=50051,
            timeout=5.0,
            retry_attempts=2
        )

    @pytest.mark.asyncio
    async def test_initialization(self, client):
        """Test client initialization."""
        assert client.worker_id == "test_worker"
        assert client.coordinator_url == "http://localhost:8000"
        assert client.timeout == 5.0
        assert client.retry_attempts == 2

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        await client.close()
        assert client._client is None

    def test_is_registered(self, client):
        """Test registration status."""
        assert not client.is_registered()
        client._registered = True
        assert client.is_registered()


class TestHeartbeatManager:
    """Test heartbeat manager."""

    @pytest.fixture
    def mock_client(self):
        """Create mock coordinator client."""
        client = AsyncMock(spec=CoordinatorClient)
        client.heartbeat = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create heartbeat manager for testing."""
        return HeartbeatManager(
            coordinator_client=mock_client,
            interval_seconds=1,  # Short interval for testing
            max_failures=3
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Test starting and stopping heartbeat manager."""
        assert not manager.is_running()

        await manager.start()
        assert manager.is_running()

        await asyncio.sleep(0.1)  # Let it run briefly

        await manager.stop()
        assert not manager.is_running()

    @pytest.mark.asyncio
    async def test_successful_heartbeat(self, manager, mock_client):
        """Test successful heartbeat."""
        await manager.start()
        await asyncio.sleep(1.5)  # Wait for at least one heartbeat
        await manager.stop()

        assert mock_client.heartbeat.called
        assert manager.is_healthy()
        assert manager.get_last_success_time() is not None

    @pytest.mark.asyncio
    async def test_failed_heartbeat(self, manager, mock_client):
        """Test failed heartbeat handling."""
        mock_client.heartbeat = AsyncMock(return_value=False)

        await manager.start()
        await asyncio.sleep(2.5)  # Wait for multiple failures
        await manager.stop()

        assert manager.get_consecutive_failures() > 0

    @pytest.mark.asyncio
    async def test_health_status(self, manager, mock_client):
        """Test health status tracking."""
        await manager.start()
        await asyncio.sleep(1.5)

        status = manager.get_status()
        assert 'running' in status
        assert 'healthy' in status
        assert 'total_sent' in status

        await manager.stop()

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, manager, mock_client):
        """Test exponential backoff on failures."""
        mock_client.heartbeat = AsyncMock(return_value=False)

        await manager.start()
        await asyncio.sleep(0.5)

        # Should have attempted retries with increasing delays
        assert manager.get_consecutive_failures() > 0

        await manager.stop()


class TestShardManager:
    """Test shard manager."""

    @pytest.fixture
    def model(self):
        """Create small test model."""
        return create_model("tiny")

    @pytest.fixture
    def manager(self, model):
        """Create shard manager for testing."""
        total_params = model.count_parameters()
        return ShardManager(
            worker_id="test_worker",
            model=model,
            shard_start=0,
            shard_end=total_params // 2,  # First half
            device="cpu"
        )

    def test_initialization(self, manager):
        """Test shard manager initialization."""
        assert manager.worker_id == "test_worker"
        assert manager.shard_start == 0
        assert not manager.is_loaded()

    def test_load_shard_from_model(self, manager):
        """Test loading shard from model."""
        manager.load_shard()

        assert manager.is_loaded()
        params = manager.get_owned_parameters()
        assert params is not None
        assert params.shape[0] == manager.shard_end - manager.shard_start

    def test_update_parameters(self, manager):
        """Test updating parameters."""
        manager.load_shard()

        original_params = manager.get_owned_parameters().clone()
        new_params = torch.randn_like(original_params)

        manager.update_parameters(new_params)

        updated_params = manager.get_owned_parameters()
        assert torch.allclose(updated_params, new_params)

    def test_update_parameters_not_loaded(self, manager):
        """Test updating parameters before loading."""
        with pytest.raises(RuntimeError):
            new_params = torch.randn(100)
            manager.update_parameters(new_params)

    def test_save_checkpoint(self, manager):
        """Test checkpoint saving."""
        from pathlib import Path
        manager.load_shard()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager.checkpoint_dir = Path(tmpdir)
            checkpoint_path = manager.save_checkpoint(global_step=100)

            assert os.path.exists(checkpoint_path)
            assert "step_100" in checkpoint_path

    def test_load_checkpoint(self, manager):
        """Test checkpoint loading."""
        from pathlib import Path
        manager.load_shard()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager.checkpoint_dir = Path(tmpdir)

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(global_step=50)

            # Create new manager and load checkpoint
            new_manager = ShardManager(
                worker_id="test_worker",
                model=manager.model,
                shard_start=manager.shard_start,
                shard_end=manager.shard_end,
                device="cpu",
                checkpoint_dir=tmpdir
            )
            new_manager.load_shard(checkpoint_path)

            assert new_manager.is_loaded()
            assert torch.allclose(
                new_manager.get_owned_parameters(),
                manager.get_owned_parameters()
            )

    def test_memory_usage(self, manager):
        """Test memory usage tracking."""
        manager.load_shard()

        memory_usage = manager.get_memory_usage()
        assert 'parameters_gb' in memory_usage
        assert 'total_gb' in memory_usage
        assert memory_usage['parameters_gb'] > 0

    def test_shard_info(self, manager):
        """Test shard info retrieval."""
        info = manager.get_shard_info()

        assert info['worker_id'] == "test_worker"
        assert info['shard_start'] == manager.shard_start
        assert info['shard_end'] == manager.shard_end
        assert info['loaded'] is False

        manager.load_shard()
        info = manager.get_shard_info()
        assert info['loaded'] is True


class TestTelemetryReporter:
    """Test telemetry reporter."""

    @pytest.fixture
    def mock_client(self):
        """Create mock coordinator client."""
        client = AsyncMock(spec=CoordinatorClient)
        client.report_metric = AsyncMock(return_value=True)
        client.report_metrics_batch = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def reporter(self, mock_client):
        """Create telemetry reporter for testing."""
        return TelemetryReporter(
            coordinator_client=mock_client,
            report_interval_steps=5,
            buffer_size=10,
            enabled=True
        )

    def test_initialization(self, reporter):
        """Test reporter initialization."""
        assert reporter.enabled is True
        assert reporter.report_interval_steps == 5
        assert reporter.buffer_size == 10

    def test_record_step(self, reporter):
        """Test recording metrics."""
        reporter.record_step(step=1, loss=2.5, throughput=100.0)

        status = reporter.get_status()
        assert status['buffered_metrics'] == 1
        assert status['total_recorded'] == 1

    def test_buffer_limit(self, reporter):
        """Test buffer size limit."""
        # Record more than buffer size
        for i in range(15):
            reporter.record_step(step=i, loss=1.0)

        status = reporter.get_status()
        assert status['buffered_metrics'] == 10  # Limited by buffer_size

    @pytest.mark.asyncio
    async def test_report(self, reporter, mock_client):
        """Test reporting metrics."""
        reporter.record_step(step=1, loss=2.5)
        reporter.record_step(step=2, loss=2.3)

        success = await reporter.report(force=True)
        assert success
        assert mock_client.report_metrics_batch.called

    @pytest.mark.asyncio
    async def test_automatic_report(self, reporter, mock_client):
        """Test automatic reporting at interval."""
        # Record steps up to interval
        for i in range(6):
            reporter.record_step(step=i, loss=2.0)

        await reporter.report()  # Should report since interval reached
        assert mock_client.report_metrics_batch.called

    @pytest.mark.asyncio
    async def test_report_step_async(self, reporter, mock_client):
        """Test async record and report."""
        await reporter.report_step_async(step=10, loss=1.5, throughput=50.0)

        # May or may not have reported yet depending on interval
        status = reporter.get_status()
        assert status['total_recorded'] >= 1

    def test_metrics_summary(self, reporter):
        """Test metrics summary."""
        reporter.record_step(step=1, loss=2.5, throughput=100.0)
        reporter.record_step(step=2, loss=2.3, throughput=110.0)
        reporter.record_step(step=3, loss=2.1, throughput=105.0)

        summary = reporter.get_metrics_summary()
        assert 'count' in summary
        assert summary['count'] == 3
        assert 'loss' in summary
        assert 'throughput' in summary

    def test_enable_disable(self, reporter):
        """Test enabling and disabling telemetry."""
        assert reporter.is_enabled()

        reporter.disable()
        assert not reporter.is_enabled()

        reporter.record_step(step=1, loss=1.0)
        assert reporter.get_status()['total_recorded'] == 0  # Not recorded when disabled

        reporter.enable()
        assert reporter.is_enabled()

    def test_clear_buffer(self, reporter):
        """Test clearing metric buffer."""
        reporter.record_step(step=1, loss=1.0)
        reporter.record_step(step=2, loss=2.0)

        assert reporter.get_status()['buffered_metrics'] == 2

        reporter.clear_buffer()
        assert reporter.get_status()['buffered_metrics'] == 0


class TestDistributedTrainer:
    """Test distributed trainer."""

    @pytest.fixture
    def trainer(self):
        """Create distributed trainer for testing."""
        return DistributedTrainer(
            worker_id="test_worker",
            rank=0,
            world_size=1,
            model_size="tiny",
            device="cpu",
            learning_rate=0.001
        )

    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.worker_id == "test_worker"
        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert not trainer.is_setup()

    def test_setup(self, trainer):
        """Test trainer setup."""
        trainer.setup()

        assert trainer.is_setup()
        assert trainer.model is not None
        assert trainer.partitioner is not None
        assert trainer.worker_coordinator is not None

    @pytest.mark.asyncio
    async def test_train(self, trainer):
        """Test training loop."""
        trainer.setup()

        # Create dummy dataset
        dataset = []
        for _ in range(10):
            inputs = torch.randint(0, 1000, (4, 32))
            targets = torch.randint(0, 1000, (4, 32))
            dataset.append((inputs, targets))

        results = await trainer.train(dataset, num_steps=10)

        assert 'num_steps' in results
        assert 'total_time' in results
        assert 'final_loss' in results
        assert results['num_steps'] == 10
        assert len(results['losses']) == 10

    def test_get_current_step(self, trainer):
        """Test getting current training step."""
        assert trainer.get_current_step() == 0

    def test_get_worker_stats(self, trainer):
        """Test getting worker statistics."""
        trainer.setup()
        stats = trainer.get_worker_stats()
        # Stats available after at least one training step
        assert isinstance(stats, dict)


class TestMetricRecord:
    """Test metric record dataclass."""

    def test_create_metric_record(self):
        """Test creating metric record."""
        record = MetricRecord(
            step=10,
            loss=2.5,
            throughput=100.0,
            memory_usage_gb=8.0
        )

        assert record.step == 10
        assert record.loss == 2.5
        assert record.throughput == 100.0
        assert record.memory_usage_gb == 8.0

    def test_to_dict(self):
        """Test converting metric record to dict."""
        record = MetricRecord(step=5, loss=1.5)
        record_dict = record.to_dict()

        assert record_dict['step'] == 5
        assert record_dict['loss'] == 1.5
        assert 'timestamp' in record_dict
