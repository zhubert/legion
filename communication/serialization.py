"""
Tensor serialization utilities for gRPC communication.

Converts between PyTorch tensors and protobuf messages for efficient
network transmission with support for compression.
"""

import torch
import numpy as np
from typing import Optional, Tuple

from communication.proto import tensor_pb2
from core.compression import INT8Quantizer


# Mapping between PyTorch dtypes and protobuf DType enum
DTYPE_TO_PROTO = {
    torch.float32: tensor_pb2.TensorProto.FLOAT32,
    torch.float16: tensor_pb2.TensorProto.FLOAT16,
    torch.bfloat16: tensor_pb2.TensorProto.BFLOAT16,
    torch.int8: tensor_pb2.TensorProto.INT8,
    torch.int32: tensor_pb2.TensorProto.INT32,
    torch.int64: tensor_pb2.TensorProto.INT64,
}

PROTO_TO_DTYPE = {v: k for k, v in DTYPE_TO_PROTO.items()}


def serialize_tensor(
    tensor: torch.Tensor,
    name: Optional[str] = None,
    compression_info: Optional[tensor_pb2.CompressionInfo] = None,
) -> tensor_pb2.TensorProto:
    """
    Convert a PyTorch tensor to protobuf format.

    Args:
        tensor: PyTorch tensor to serialize
        name: Optional identifier for the tensor
        compression_info: Optional compression metadata

    Returns:
        TensorProto message ready for gRPC transmission
    """
    # Ensure tensor is contiguous for efficient serialization
    tensor = tensor.contiguous()

    # Convert to numpy and serialize to bytes
    tensor_np = tensor.cpu().numpy()
    tensor_bytes = tensor_np.tobytes()

    # Create protobuf message
    proto = tensor_pb2.TensorProto()
    proto.shape.extend(tensor.shape)
    proto.dtype = DTYPE_TO_PROTO[tensor.dtype]
    proto.data = tensor_bytes

    if name:
        proto.name = name

    if compression_info:
        proto.compression.CopyFrom(compression_info)

    return proto


def deserialize_tensor(
    proto: tensor_pb2.TensorProto, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a protobuf TensorProto back to PyTorch tensor.

    Args:
        proto: TensorProto message from gRPC
        device: Target device for the tensor (CPU or CUDA)

    Returns:
        PyTorch tensor reconstructed from protobuf
    """
    # Get dtype
    dtype = PROTO_TO_DTYPE[proto.dtype]

    # Convert bytes to numpy array
    np_dtype = torch_to_numpy_dtype(dtype)
    tensor_np = np.frombuffer(proto.data, dtype=np_dtype)

    # Reshape to original shape
    shape = tuple(proto.shape)
    tensor_np = tensor_np.reshape(shape)

    # Copy to make writable before converting to PyTorch
    # frombuffer creates read-only arrays, causing warnings
    tensor = torch.from_numpy(tensor_np.copy())

    # Move to target device if specified
    if device:
        tensor = tensor.to(device)

    return tensor


def torch_to_numpy_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """Convert PyTorch dtype to numpy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.uint16,  # bfloat16 stored as uint16 in numpy
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }
    return mapping[torch_dtype]


def create_compression_info(
    compression_type: str,
    scale: Optional[float] = None,
    zero_point: Optional[float] = None,
    k: Optional[int] = None,
    indices: Optional[list] = None,
    original_shape: Optional[tuple] = None,
    original_dtype: Optional[torch.dtype] = None,
) -> tensor_pb2.CompressionInfo:
    """
    Create compression metadata for a tensor.

    Args:
        compression_type: One of 'none', 'int8', 'topk', 'onebit'
        scale: For INT8 quantization
        zero_point: For INT8 quantization
        k: Number of elements kept for TopK
        indices: Indices of kept elements for TopK
        original_shape: Shape before compression
        original_dtype: Dtype before compression

    Returns:
        CompressionInfo protobuf message
    """
    info = tensor_pb2.CompressionInfo()

    # Set compression type
    type_map = {
        "none": tensor_pb2.CompressionInfo.NONE,
        "int8": tensor_pb2.CompressionInfo.INT8_QUANTIZE,
        "topk": tensor_pb2.CompressionInfo.TOPK_SPARSE,
        "onebit": tensor_pb2.CompressionInfo.ONEBIT,
    }
    info.type = type_map[compression_type.lower()]

    # Set INT8 quantization params
    if scale is not None:
        info.scale = scale
    if zero_point is not None:
        info.zero_point = zero_point

    # Set TopK params
    if k is not None:
        info.k = k
    if indices is not None:
        info.indices.extend(indices)

    # Set original tensor metadata
    if original_shape is not None:
        info.original_shape.extend(original_shape)
    if original_dtype is not None:
        info.original_dtype = DTYPE_TO_PROTO[original_dtype]

    return info


def chunk_tensor(
    tensor: torch.Tensor, chunk_size_mb: int = 10, name: Optional[str] = None
) -> list[tensor_pb2.TensorChunk]:
    """
    Split a large tensor into chunks for streaming.

    Args:
        tensor: Tensor to chunk
        chunk_size_mb: Size of each chunk in megabytes
        name: Optional tensor name

    Returns:
        List of TensorChunk messages
    """
    # Serialize tensor to bytes
    tensor = tensor.contiguous()
    tensor_np = tensor.cpu().numpy()
    tensor_bytes = tensor_np.tobytes()

    # Calculate chunk size in bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # Split into chunks
    chunks = []
    total_chunks = (len(tensor_bytes) + chunk_size_bytes - 1) // chunk_size_bytes

    for i in range(total_chunks):
        start = i * chunk_size_bytes
        end = min((i + 1) * chunk_size_bytes, len(tensor_bytes))
        chunk_data = tensor_bytes[start:end]

        chunk = tensor_pb2.TensorChunk()
        chunk.chunk_id = i
        chunk.total_chunks = total_chunks
        chunk.data = chunk_data

        # Include metadata only in first chunk
        if i == 0:
            chunk.dtype = DTYPE_TO_PROTO[tensor.dtype]
            chunk.shape.extend(tensor.shape)
            if name:
                chunk.name = name

        chunks.append(chunk)

    return chunks


def reassemble_chunks(chunks: list[tensor_pb2.TensorChunk]) -> torch.Tensor:
    """
    Reassemble a tensor from streamed chunks.

    Args:
        chunks: List of TensorChunk messages

    Returns:
        Reconstructed PyTorch tensor
    """
    # Sort chunks by ID (in case they arrived out of order)
    chunks = sorted(chunks, key=lambda c: c.chunk_id)

    # Concatenate chunk data
    tensor_bytes = b"".join(chunk.data for chunk in chunks)

    # Get metadata from first chunk
    first_chunk = chunks[0]
    dtype = PROTO_TO_DTYPE[first_chunk.dtype]
    shape = tuple(first_chunk.shape)

    # Reconstruct tensor
    np_dtype = torch_to_numpy_dtype(dtype)
    tensor_np = np.frombuffer(tensor_bytes, dtype=np_dtype)
    tensor_np = tensor_np.reshape(shape)

    # Copy to make writable before converting to PyTorch
    return torch.from_numpy(tensor_np.copy())


def serialize_tensor_compressed(
    tensor: torch.Tensor,
    name: Optional[str] = None,
    compress: bool = True
) -> tensor_pb2.TensorProto:
    """
    Serialize tensor with optional INT8 compression.

    Args:
        tensor: PyTorch tensor to serialize
        name: Optional identifier for the tensor
        compress: Whether to apply INT8 compression (default: True)

    Returns:
        TensorProto message with compressed data if enabled
    """
    if not compress or tensor.dtype != torch.float32:
        # No compression or already compressed dtype
        return serialize_tensor(tensor, name=name)

    # Compress with INT8 quantization
    quantized, scale = INT8Quantizer.quantize(tensor)

    # Create compression metadata
    compression_info = create_compression_info(
        compression_type="int8",
        scale=scale.item(),
        original_shape=tuple(tensor.shape),
        original_dtype=tensor.dtype
    )

    # Serialize the quantized tensor
    return serialize_tensor(quantized, name=name, compression_info=compression_info)


def deserialize_tensor_compressed(
    proto: tensor_pb2.TensorProto,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Deserialize tensor and decompress if needed.

    Args:
        proto: TensorProto message from gRPC
        device: Target device for the tensor

    Returns:
        Decompressed PyTorch tensor
    """
    # Check if compression was used
    if proto.compression.type == tensor_pb2.CompressionInfo.NONE:
        # No compression
        return deserialize_tensor(proto, device=device)

    elif proto.compression.type == tensor_pb2.CompressionInfo.INT8_QUANTIZE:
        # INT8 compression - decompress
        quantized = deserialize_tensor(proto, device=device)
        scale = torch.tensor(proto.compression.scale)

        # Dequantize
        tensor = INT8Quantizer.dequantize(quantized, scale)

        # Restore original shape if needed
        if proto.compression.original_shape:
            original_shape = tuple(proto.compression.original_shape)
            tensor = tensor.view(original_shape)

        # Move to target device
        if device:
            tensor = tensor.to(device)

        return tensor

    else:
        raise ValueError(f"Unsupported compression type: {proto.compression.type}")
