"""
Tests for worker configuration device auto-detection.
"""

import pytest
from unittest.mock import patch, MagicMock
from worker.config import get_best_device, WorkerConfig


class TestDeviceAutoDetection:
    """Test device auto-detection logic."""

    def test_get_best_device_cuda_available(self):
        """Test that CUDA is preferred when available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_best_device()
            assert device == "cuda"

    def test_get_best_device_mps_available(self):
        """Test that MPS is chosen when CUDA not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict('sys.modules', {'torch': mock_torch}):
            device = get_best_device()
            assert device == "mps"

    def test_get_best_device_cpu_fallback(self):
        """Test that CPU is used when no accelerators available."""
        # Mock torch module with both CUDA and MPS unavailable
        with patch('torch.cuda.is_available', return_value=False):
            # Create a mock for MPS that raises AttributeError (not available)
            with patch('torch.backends', MagicMock(spec=[])):
                device = get_best_device()
                assert device == "cpu"

    def test_get_best_device_no_torch(self):
        """Test CPU fallback when torch not available."""
        with patch.dict('sys.modules', {'torch': None}):
            device = get_best_device()
            assert device == "cpu"

    def test_worker_config_auto_detects_device(self):
        """Test that WorkerConfig auto-detects device."""
        config = WorkerConfig()
        assert config.device in ["cuda", "mps", "cpu"]

    def test_worker_config_manual_device_override(self):
        """Test that manual device specification works."""
        config = WorkerConfig(device="cpu")
        assert config.device == "cpu"

    def test_worker_config_mps_sets_gpu_name(self):
        """Test that MPS device gets proper GPU name."""
        with patch('worker.config.get_best_device', return_value="mps"), \
             patch('torch.backends.mps.is_available', return_value=True):
            config = WorkerConfig()
            if config.device == "mps":
                assert config.gpu_name == "Apple Silicon (MPS)"

    def test_worker_config_cuda_fallback_if_unavailable(self):
        """Test that config falls back to CPU if CUDA requested but unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            config = WorkerConfig(device="cuda")
            assert config.device == "cpu"
