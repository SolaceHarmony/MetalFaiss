"""
ops.py - MLX-specific operations for MetalFaiss

This module provides MLX implementations of core operations needed by MetalFaiss.
If we want to port to another backend, we would create a similar module for that
backend (e.g., faisstorch/ops.py for PyTorch).

Operations are organized into categories:
- Array ops: Creation, manipulation, indexing
- Math ops: Basic arithmetic, reduction
- Matrix ops: Linear algebra operations
- Distance ops: Vector distance computations
- Binary ops: Binary vector operations
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple, Union, Optional
from enum import Enum

class Device(Enum):
    """Device types for array placement."""
    CPU = "cpu"
    GPU = "gpu"

# Array Creation and Manipulation Ops

def array(
    data: Union[List, np.ndarray],
    dtype: Optional[str] = None,
    device: Device = Device.CPU
) -> mx.array:
    """Create array from data.
    
    Args:
        data: Input data
        dtype: Optional dtype (inferred if None)
        device: Device to place array on
        
    Returns:
        MLX array on specified device
    """
    arr = mx.array(data, dtype=dtype)
    # TODO: Add device placement when MLX supports it
    return arr

def zeros(
    shape: Tuple[int, ...],
    dtype: str = "float32",
    device: Device = Device.CPU
) -> mx.array:
    """Create array of zeros.
    
    Args:
        shape: Array shape
        dtype: Data type
        device: Device to place array on
        
    Returns:
        Array of zeros
    """
    arr = mx.zeros(shape, dtype=dtype)
    # TODO: Add device placement
    return arr

def ones(
    shape: Tuple[int, ...],
    dtype: str = "float32",
    device: Device = Device.CPU
) -> mx.array:
    """Create array of ones."""
    arr = mx.ones(shape, dtype=dtype)
    # TODO: Add device placement
    return arr

def arange(
    start: int,
    stop: Optional[int] = None,
    step: int = 1,
    dtype: str = "int32",
    device: Device = Device.CPU
) -> mx.array:
    """Create range array."""
    arr = mx.arange(start, stop, step, dtype=dtype)
    # TODO: Add device placement
    return arr

def concatenate(
    arrays: List[mx.array],
    axis: int = 0,
    device: Device = Device.CPU
) -> mx.array:
    """Concatenate arrays."""
    arr = mx.concatenate(arrays, axis=axis)
    # TODO: Add device placement
    return arr

# Math Ops

def sum(
    x: mx.array,
    axis: Optional[int] = None,
    keepdims: bool = False
) -> mx.array:
    """Array sum reduction."""
    return mx.sum(x, axis=axis, keepdims=keepdims)

def mean(
    x: mx.array,
    axis: Optional[int] = None,
    keepdims: bool = False
) -> mx.array:
    """Array mean reduction."""
    return mx.mean(x, axis=axis, keepdims=keepdims)

def min(
    x: mx.array,
    axis: Optional[int] = None,
    keepdims: bool = False
) -> mx.array:
    """Array minimum reduction."""
    return mx.min(x, axis=axis, keepdims=keepdims)

def max(
    x: mx.array,
    axis: Optional[int] = None,
    keepdims: bool = False
) -> mx.array:
    """Array maximum reduction."""
    return mx.max(x, axis=axis, keepdims=keepdims)

# Matrix Ops

def matmul(a: mx.array, b: mx.array) -> mx.array:
    """Matrix multiplication."""
    return mx.matmul(a, b)

def transpose(x: mx.array, axes: Optional[Tuple[int, ...]] = None) -> mx.array:
    """Array transpose."""
    return mx.transpose(x, axes)

# Distance Ops

def l2_distances(x: mx.array, y: mx.array) -> mx.array:
    """Compute pairwise L2 distances.
    
    Args:
        x: First vectors (n, d)
        y: Second vectors (m, d)
        
    Returns:
        Distance matrix (n, m)
    """
    # Compute (a-b)^2 = a^2 + b^2 - 2ab
    xx = sum(x * x, axis=1, keepdims=True)  # (n, 1)
    yy = sum(y * y, axis=1)  # (m,)
    xy = matmul(x, transpose(y))  # (n, m)
    return xx + yy - 2 * xy

def cosine_distances(x: mx.array, y: mx.array) -> mx.array:
    """Compute pairwise cosine distances.
    
    Args:
        x: First vectors (n, d)
        y: Second vectors (m, d)
        
    Returns:
        Distance matrix (n, m)
    """
    # Normalize vectors
    x_norm = sum(x * x, axis=1, keepdims=True) ** 0.5
    y_norm = sum(y * y, axis=1, keepdims=True) ** 0.5
    x = x / x_norm
    y = y / y_norm
    
    # Compute 1 - cos(theta) = 1 - x·y
    return 1 - matmul(x, transpose(y))

def hamming_distances(x: mx.array, y: mx.array) -> mx.array:
    """Compute pairwise Hamming distances.
    
    Args:
        x: First binary vectors (n, d) of uint8
        y: Second binary vectors (m, d) of uint8
        
    Returns:
        Distance matrix (n, m) of uint32
    """
    # Create lookup table for Hamming weight
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = bin(i).count('1')
    table = array(table, dtype="uint8")
    
    # Compute XOR then lookup Hamming weights
    xor = x[:, None, :] ^ y[None, :, :]  # (n, m, d)
    return sum(table[xor], axis=2, dtype="uint32")

# Binary Ops

def binary_and(x: mx.array, y: mx.array) -> mx.array:
    """Bitwise AND."""
    return x & y

def binary_or(x: mx.array, y: mx.array) -> mx.array:
    """Bitwise OR."""
    return x | y

def binary_xor(x: mx.array, y: mx.array) -> mx.array:
    """Bitwise XOR."""
    return x ^ y

def binary_not(x: mx.array) -> mx.array:
    """Bitwise NOT."""
    return ~x

def popcount(x: mx.array) -> mx.array:
    """Count number of 1 bits in each element.
    
    Args:
        x: Input array of uint8
        
    Returns:
        Array with same shape containing bit counts
    """
    # Use lookup table for uint8
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = bin(i).count('1')
    table = array(table, dtype="uint8")
    return table[x]

# Device Management

def to_device(x: mx.array, device: Device) -> mx.array:
    """Move array to specified device.
    
    Args:
        x: Input array
        device: Target device
        
    Returns:
        Array on target device
    """
    # TODO: Implement when MLX supports explicit device placement
    return x

def get_device(x: mx.array) -> Device:
    """Get device of array.
    
    Args:
        x: Input array
        
    Returns:
        Device containing array
    """
    # TODO: Implement when MLX supports device queries
    return Device.CPU