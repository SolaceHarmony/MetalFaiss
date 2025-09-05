"""
gpu_ops.py - GPU-optimized operations for MetalFaiss

This module provides GPU implementations of core operations using MLX.
Operations are optimized for GPU execution and include memory management
utilities.

Note: Currently uses MLX's automatic device placement, but prepared for
more explicit control when MLX adds those features.
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple, Union, Optional
from enum import Enum
from dataclasses import dataclass
from .ops import Device

@dataclass
class GpuResources:
    """Track GPU memory and resources.
    
    This will be expanded when MLX adds more explicit GPU control.
    Currently mostly a placeholder for future functionality.
    """
    device_id: int = 0
    max_memory: Optional[int] = None
    current_memory: int = 0
    
    def __post_init__(self):
        """Initialize GPU device."""
        # TODO: Add explicit device initialization when MLX supports it
        pass
        
    def allocate(self, size: int) -> bool:
        """Track memory allocation.
        
        Args:
            size: Bytes to allocate
            
        Returns:
            Whether allocation succeeded
        """
        if self.max_memory is not None:
            if self.current_memory + size > self.max_memory:
                return False
        self.current_memory += size
        return True
        
    def free(self, size: int) -> None:
        """Track memory deallocation."""
        self.current_memory = max(0, self.current_memory - size)
        
    def reset(self) -> None:
        """Reset memory tracking."""
        self.current_memory = 0

# Global GPU resources
DEFAULT_GPU = GpuResources()

# GPU Array Operations

def to_gpu(x: Union[mx.array, np.ndarray, List]) -> mx.array:
    """Move array to GPU.
    
    Args:
        x: Input array-like object
        
    Returns:
        MLX array on GPU
    """
    if isinstance(x, mx.array):
        # TODO: Add explicit device move when MLX supports it
        return x
    elif isinstance(x, np.ndarray):
        return mx.array(x)
    else:
        return mx.array(x)

def from_gpu(x: mx.array) -> np.ndarray:
    """Move array from GPU to CPU.
    
    Args:
        x: Input MLX array
        
    Returns:
        NumPy array on CPU
    """
    # TODO: Add explicit device move when MLX supports it
    return x.numpy()

# GPU Matrix Operations

def gpu_matmul(a: mx.array, b: mx.array) -> mx.array:
    """GPU-optimized matrix multiplication.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Returns:
        Matrix product on GPU
    """
    # TODO: Add optimized implementation when MLX adds GPU-specific ops
    return mx.matmul(a, b)

def gpu_l2_distances(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized L2 distance computation.
    
    Args:
        x: First vectors (n, d)
        y: Second vectors (m, d)
        
    Returns:
        Distance matrix (n, m) on GPU
    """
    # Compute (a-b)^2 = a^2 + b^2 - 2ab efficiently on GPU
    xx = mx.sum(x * x, axis=1, keepdims=True)
    yy = mx.sum(y * y, axis=1)
    xy = gpu_matmul(x, mx.transpose(y))
    return xx + yy - 2 * xy

def gpu_cosine_distances(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized cosine distance computation.
    
    Args:
        x: First vectors (n, d)
        y: Second vectors (m, d)
        
    Returns:
        Distance matrix (n, m) on GPU
    """
    # Normalize and compute dot products efficiently
    x_norm = mx.sum(x * x, axis=1, keepdims=True) ** 0.5
    y_norm = mx.sum(y * y, axis=1, keepdims=True) ** 0.5
    x = x / x_norm
    y = y / y_norm
    return 1 - gpu_matmul(x, mx.transpose(y))

# GPU Binary Operations

def gpu_hamming_distances(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized Hamming distance computation.
    
    Args:
        x: First binary vectors (n, d) of uint8
        y: Second binary vectors (m, d) of uint8
        
    Returns:
        Distance matrix (n, m) of uint32 on GPU
    """
    # Create lookup table for Hamming weight
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = bin(i).count('1')
    table = mx.array(table, dtype=mx.uint8)
    
    # Compute XOR then lookup Hamming weights
    # TODO: Add optimized implementation when MLX adds GPU-specific ops
    xor = x[:, None, :] ^ y[None, :, :]
    return mx.sum(table[xor], axis=2, dtype=mx.uint32)

def gpu_binary_and(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized binary AND."""
    # TODO: Add optimized implementation
    return x & y

def gpu_binary_or(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized binary OR."""
    # TODO: Add optimized implementation
    return x | y

def gpu_binary_xor(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized binary XOR."""
    # TODO: Add optimized implementation
    return x ^ y

def gpu_popcount(x: mx.array) -> mx.array:
    """GPU-optimized population count.
    
    Args:
        x: Input array of uint8
        
    Returns:
        Array with same shape containing bit counts
    """
    # TODO: Add optimized implementation
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = bin(i).count('1')
    table = mx.array(table, dtype=mx.uint8)
    return table[x]

# Memory Management

class GpuMemoryManager:
    """Manage GPU memory allocations.
    
    This will be expanded when MLX adds more explicit memory control.
    Currently mostly tracks allocations for monitoring.
    """
    
    def __init__(self, resources: Optional[GpuResources] = None):
        """Initialize manager.
        
        Args:
            resources: GPU resources to manage
        """
        self.resources = resources or DEFAULT_GPU
        
    def __enter__(self):
        """Enter context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.resources.reset()
        
    def alloc(self, shape: Tuple[int, ...], dtype: str = "float32") -> mx.array:
        """Allocate GPU array.
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            Allocated array on GPU
            
        Raises:
            MemoryError: If allocation would exceed memory limit
        """
        # Compute size in bytes
        element_size = {
            "float32": 4,
            "float64": 8,
            "int32": 4,
            "int64": 8,
            "uint8": 1
        }.get(dtype, 4)
        size = np.prod(shape) * element_size
        
        if not self.resources.allocate(size):
            raise MemoryError("GPU memory limit exceeded")
            
        return mx.zeros(shape, dtype=dtype)
        
    def free(self, arr: mx.array) -> None:
        """Free GPU array.
        
        Args:
            arr: Array to free
        """
        # Compute size in bytes
        element_size = {
            mx.float32: 4,
            mx.float64: 8,
            mx.int32: 4,
            mx.int64: 8,
            mx.uint8: 1
        }.get(arr.dtype, 4)
        size = np.prod(arr.shape) * element_size
        
        self.resources.free(size)
        # Actual freeing handled by MLX