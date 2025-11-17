"""
Gradient Compression for Bandwidth Reduction

Implements various compression techniques to reduce communication overhead:
- INT8 quantization (4x compression for FP32)
- Top-K sparsification
- Error feedback/compensation

Future: 1-bit Adam (32x compression)
"""

import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics about compression performance"""
    original_size: int  # Number of elements
    compressed_size: int  # Number of bytes after compression
    compression_ratio: float
    error: float  # L2 norm of compression error


class INT8Quantizer:
    """
    INT8 Quantization with per-tensor scaling.

    Compresses FP32 tensors to INT8, achieving 4x compression.
    Uses a single scale factor per tensor.
    """

    @staticmethod
    def quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize FP32 tensor to INT8.

        Args:
            tensor: Input tensor (FP32)

        Returns:
            quantized: INT8 tensor
            scale: Scale factor for dequantization
        """
        # Compute scale: max absolute value / 127
        max_val = tensor.abs().max()
        scale = max_val / 127.0

        # Avoid division by zero
        if scale == 0:
            scale = torch.tensor(1.0)

        # Quantize: scale, round, clamp to [-128, 127]
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)

        return quantized, scale

    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT8 tensor back to FP32.

        Args:
            quantized: INT8 tensor
            scale: Scale factor from quantization

        Returns:
            FP32 tensor
        """
        return quantized.to(torch.float32) * scale

    @staticmethod
    def compress_size(tensor: torch.Tensor) -> int:
        """Calculate size in bytes after INT8 compression"""
        # 1 byte per element + 4 bytes for FP32 scale
        return tensor.numel() + 4


class BlockQuantizer:
    """
    Block-based INT8 quantization for better accuracy.

    Divides tensor into blocks and quantizes each independently.
    This preserves more information than per-tensor quantization.
    """

    def __init__(self, block_size: int = 1024):
        """
        Args:
            block_size: Number of elements per block
        """
        self.block_size = block_size

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor using block-based approach.

        Args:
            tensor: Input tensor (FP32)

        Returns:
            quantized: INT8 tensor
            scales: Scale factors per block
        """
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()

        # Pad to multiple of block_size
        numel = flat_tensor.numel()
        num_blocks = (numel + self.block_size - 1) // self.block_size
        padded_size = num_blocks * self.block_size

        if padded_size > numel:
            padding = torch.zeros(padded_size - numel, dtype=tensor.dtype, device=tensor.device)
            flat_tensor = torch.cat([flat_tensor, padding])

        # Reshape into blocks
        blocks = flat_tensor.view(num_blocks, self.block_size)

        # Quantize each block
        quantized_blocks = []
        scales = []

        for block in blocks:
            max_val = block.abs().max()
            scale = max_val / 127.0

            if scale == 0:
                scale = torch.tensor(1.0, dtype=torch.float32)

            quant_block = (block / scale).round().clamp(-128, 127).to(torch.int8)
            quantized_blocks.append(quant_block)
            scales.append(scale)

        # Concatenate results
        quantized = torch.cat(quantized_blocks)
        scales = torch.stack(scales)

        # Store original shape for reconstruction
        return quantized, scales

    def dequantize(self, quantized: torch.Tensor, scales: torch.Tensor,
                   original_shape: torch.Size) -> torch.Tensor:
        """
        Dequantize block-quantized tensor.

        Args:
            quantized: INT8 tensor
            scales: Scale factors per block
            original_shape: Original tensor shape

        Returns:
            FP32 tensor with original shape
        """
        num_blocks = scales.numel()
        blocks = quantized.view(num_blocks, self.block_size)

        # Dequantize each block
        dequant_blocks = []
        for i, block in enumerate(blocks):
            dequant_block = block.to(torch.float32) * scales[i]
            dequant_blocks.append(dequant_block)

        # Concatenate and reshape
        result = torch.cat(dequant_blocks)

        # Remove padding
        original_numel = torch.Size(original_shape).numel()
        result = result[:original_numel]

        return result.view(original_shape)


class TopKSparsifier:
    """
    Top-K gradient sparsification.

    Only sends the K largest gradients (by magnitude).
    Other gradients are accumulated locally (error feedback).
    """

    def __init__(self, k: float = 0.01, use_error_feedback: bool = True):
        """
        Args:
            k: Fraction of gradients to keep (0.01 = 1%)
            use_error_feedback: Whether to use error feedback
        """
        self.k = k
        self.use_error_feedback = use_error_feedback
        self.error_buffer = {}  # Accumulated errors per parameter

    def sparsify(self, tensor: torch.Tensor, name: str = "default") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparsify tensor by keeping only top-K values.

        Args:
            tensor: Input tensor
            name: Parameter name (for error tracking)

        Returns:
            values: Top-K values
            indices: Indices of top-K values
            shape: Original tensor shape
        """
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()

        # Add error feedback if enabled
        if self.use_error_feedback and name in self.error_buffer:
            flat_tensor = flat_tensor + self.error_buffer[name]

        # Select top-K by magnitude
        k_count = max(1, int(flat_tensor.numel() * self.k))
        abs_values = flat_tensor.abs()
        _, indices = torch.topk(abs_values, k_count)

        # Extract top-K values
        values = flat_tensor[indices]

        # Store error feedback (dropped gradients)
        if self.use_error_feedback:
            error = flat_tensor.clone()
            error[indices] = 0  # Zero out transmitted values
            self.error_buffer[name] = error

        return values, indices, torch.tensor(original_shape)

    def desparsify(self, values: torch.Tensor, indices: torch.Tensor,
                   shape: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct sparse tensor.

        Args:
            values: Top-K values
            indices: Indices of top-K values
            shape: Original tensor shape

        Returns:
            Reconstructed tensor (sparse)
        """
        # Create zero tensor
        flat_shape = torch.Size(shape.tolist()).numel()
        result = torch.zeros(flat_shape, dtype=values.dtype, device=values.device)

        # Fill in top-K values
        result[indices] = values

        # Reshape
        return result.view(shape.tolist())

    def compress_size(self, tensor: torch.Tensor) -> int:
        """Calculate size in bytes after sparsification"""
        k_count = max(1, int(tensor.numel() * self.k))
        # 4 bytes per FP32 value + 4 bytes per int32 index
        return k_count * 8


class CompressionManager:
    """
    Manages gradient compression strategies.

    Provides unified interface for different compression techniques.
    """

    def __init__(self, method: str = "int8", **kwargs):
        """
        Args:
            method: Compression method ('int8', 'block', 'topk', 'none')
            **kwargs: Method-specific arguments
        """
        self.method = method

        if method == "int8":
            self.compressor = INT8Quantizer()
        elif method == "block":
            block_size = kwargs.get('block_size', 1024)
            self.compressor = BlockQuantizer(block_size)
        elif method == "topk":
            k = kwargs.get('k', 0.01)
            self.compressor = TopKSparsifier(k)
        elif method == "none":
            self.compressor = None
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def compress(self, tensor: torch.Tensor, name: str = "default") -> Tuple:
        """
        Compress tensor.

        Args:
            tensor: Input tensor
            name: Parameter name

        Returns:
            Compressed representation (method-specific)
        """
        if self.method == "none":
            return (tensor,)

        elif self.method == "int8":
            quantized, scale = self.compressor.quantize(tensor)
            return (quantized, scale)

        elif self.method == "block":
            quantized, scales = self.compressor.quantize(tensor)
            return (quantized, scales, torch.tensor(tensor.shape))

        elif self.method == "topk":
            values, indices, shape = self.compressor.sparsify(tensor, name)
            return (values, indices, shape)

    def decompress(self, compressed: Tuple) -> torch.Tensor:
        """
        Decompress tensor.

        Args:
            compressed: Compressed representation

        Returns:
            Decompressed tensor
        """
        if self.method == "none":
            return compressed[0]

        elif self.method == "int8":
            quantized, scale = compressed
            return self.compressor.dequantize(quantized, scale)

        elif self.method == "block":
            quantized, scales, shape = compressed
            return self.compressor.dequantize(quantized, scales, shape)

        elif self.method == "topk":
            values, indices, shape = compressed
            return self.compressor.desparsify(values, indices, shape)

    def get_stats(self, original: torch.Tensor, compressed: Tuple) -> CompressionStats:
        """
        Calculate compression statistics.

        Args:
            original: Original tensor
            compressed: Compressed representation

        Returns:
            CompressionStats object
        """
        original_size = original.numel() * 4  # FP32 = 4 bytes

        # Calculate compressed size
        if self.method == "none":
            compressed_size = original_size
        elif self.method == "int8":
            compressed_size = compressed[0].numel() + 4
        elif self.method == "block":
            compressed_size = compressed[0].numel() + compressed[1].numel() * 4
        elif self.method == "topk":
            compressed_size = compressed[0].numel() * 4 + compressed[1].numel() * 4

        compression_ratio = original_size / compressed_size

        # Calculate error
        decompressed = self.decompress(compressed)
        error = (original - decompressed).norm().item()

        return CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            error=error
        )


if __name__ == "__main__":
    print("Testing Compression Methods\n")

    # Create test tensor
    test_tensor = torch.randn(10000)

    methods = ["int8", "block", "topk", "none"]

    for method in methods:
        print(f"\nTesting {method.upper()} compression:")

        if method == "topk":
            manager = CompressionManager(method, k=0.1)  # 10% sparsity
        else:
            manager = CompressionManager(method)

        # Compress
        compressed = manager.compress(test_tensor)

        # Decompress
        decompressed = manager.decompress(compressed)

        # Get stats
        stats = manager.get_stats(test_tensor, compressed)

        print(f"  Original size: {stats.original_size:,} bytes")
        print(f"  Compressed size: {stats.compressed_size:,} bytes")
        print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
        print(f"  Reconstruction error (L2): {stats.error:.6f}")

        # Verify shape
        assert decompressed.shape == test_tensor.shape, "Shape mismatch!"

    print("\n✓ All compression tests passed!")
