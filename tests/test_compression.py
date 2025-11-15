"""
Tests for gradient compression methods
"""

import pytest
import torch
from sim.compression import (
    INT8Quantizer,
    BlockQuantizer,
    TopKSparsifier,
    CompressionManager,
    CompressionStats
)


class TestINT8Quantizer:
    """Test INT8 quantization"""

    def test_quantize_dequantize(self):
        """Test that quantize/dequantize round-trip works"""
        tensor = torch.randn(1000)

        quantized, scale = INT8Quantizer.quantize(tensor)

        assert quantized.dtype == torch.int8
        assert scale.dtype == torch.float32

        dequantized = INT8Quantizer.dequantize(quantized, scale)

        assert dequantized.shape == tensor.shape
        # Should have some error but be close
        assert torch.allclose(dequantized, tensor, atol=0.1)

    def test_quantize_zeros(self):
        """Test quantizing tensor of zeros"""
        tensor = torch.zeros(100)

        quantized, scale = INT8Quantizer.quantize(tensor)
        dequantized = INT8Quantizer.dequantize(quantized, scale)

        assert torch.allclose(dequantized, tensor)

    def test_compression_ratio(self):
        """Test that INT8 achieves ~4x compression"""
        tensor = torch.randn(1000)
        compressed_size = INT8Quantizer.compress_size(tensor)
        original_size = tensor.numel() * 4  # FP32

        ratio = original_size / compressed_size
        assert ratio > 3.5  # Should be close to 4x


class TestBlockQuantizer:
    """Test block-based quantization"""

    def test_block_quantize_dequantize(self):
        """Test block quantization round-trip"""
        tensor = torch.randn(10000)
        quantizer = BlockQuantizer(block_size=1024)

        quantized, scales = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized, scales, tensor.shape)

        assert dequantized.shape == tensor.shape
        assert torch.allclose(dequantized, tensor, atol=0.1)

    def test_block_quantize_preserves_shape(self):
        """Test that block quantization handles arbitrary shapes"""
        tensor = torch.randn(3, 4, 5)
        quantizer = BlockQuantizer(block_size=10)

        quantized, scales = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized, scales, tensor.shape)

        assert dequantized.shape == tensor.shape


class TestTopKSparsifier:
    """Test Top-K sparsification"""

    def test_topk_sparsify(self):
        """Test Top-K sparsification"""
        tensor = torch.randn(1000)
        sparsifier = TopKSparsifier(k=0.1, use_error_feedback=False)

        values, indices, shape = sparsifier.sparsify(tensor)

        # Should keep 10% of values
        assert values.numel() == 100
        assert indices.numel() == 100

    def test_topk_desparsify(self):
        """Test Top-K desparsification"""
        tensor = torch.randn(1000)
        sparsifier = TopKSparsifier(k=0.1, use_error_feedback=False)

        values, indices, shape = sparsifier.sparsify(tensor)
        reconstructed = sparsifier.desparsify(values, indices, shape)

        assert reconstructed.shape == tensor.shape
        # Most values should be zero (90%)
        assert (reconstructed == 0).sum() >= 900

    def test_topk_error_feedback(self):
        """Test that error feedback accumulates"""
        tensor = torch.randn(1000)
        sparsifier = TopKSparsifier(k=0.1, use_error_feedback=True)

        # First sparsification
        values1, indices1, shape1 = sparsifier.sparsify(tensor, name="test")

        # Second sparsification should use error feedback
        values2, indices2, shape2 = sparsifier.sparsify(tensor, name="test")

        # Error buffer should exist
        assert "test" in sparsifier.error_buffer


class TestCompressionManager:
    """Test unified compression manager"""

    @pytest.mark.parametrize("method", ["int8", "block", "topk", "none"])
    def test_compression_methods(self, method):
        """Test all compression methods"""
        tensor = torch.randn(1000)

        if method == "topk":
            manager = CompressionManager(method, k=0.1)
        else:
            manager = CompressionManager(method)

        # Compress
        compressed = manager.compress(tensor)
        assert compressed is not None

        # Decompress
        decompressed = manager.decompress(compressed)
        assert decompressed.shape == tensor.shape

    def test_compression_stats(self):
        """Test compression statistics calculation"""
        tensor = torch.randn(1000)
        manager = CompressionManager("int8")

        compressed = manager.compress(tensor)
        stats = manager.get_stats(tensor, compressed)

        assert isinstance(stats, CompressionStats)
        assert stats.compression_ratio > 1.0
        assert stats.original_size > stats.compressed_size

    def test_none_compression(self):
        """Test that 'none' method doesn't compress"""
        tensor = torch.randn(1000)
        manager = CompressionManager("none")

        compressed = manager.compress(tensor)
        decompressed = manager.decompress(compressed)

        assert torch.equal(tensor, decompressed)

    def test_invalid_method(self):
        """Test that invalid method raises error"""
        with pytest.raises(ValueError):
            CompressionManager("invalid_method")
