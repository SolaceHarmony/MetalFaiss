"""
GPU ops (MLX): placement, math, and batching helpers (plain-speech)

Overview
- Uses MLX vectorized ops for core math; kernels are available elsewhere.
- Placement is primarily via MLX defaults; when MLX exposes explicit array moves,
  these helpers will adopt them.

Notes
- Avoid surprising device jumps by keeping a stable default device and streams
  during hot loops. Use `mx.set_default_device(mx.gpu)` at program start on GPU.
"""

import mlx.core as mx
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from .ops import Device
from .device_guard import require_gpu

# Graceful compile decorator if available
try:
    compile_fn = mx.compile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def compile_fn(f):
        return f

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
        """Basic parameter validation for GPU-only setup."""
        if self.device_id != 0:
            raise ValueError("Only device_id=0 is supported in GPU-only mode")
        if self.max_memory is not None and self.max_memory <= 0:
            raise ValueError("max_memory must be positive if provided")
        
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

# Professional aliases (GPU-only project): prefer neutral names
class Resources(GpuResources):
    pass

# GPU Array Operations

__all__ = [
    'GpuResources', 'GpuMemoryManager', 'Resources', 'MemoryManager',
    'matmul', 'l2_distances', 'l2_distances_chunked', 'cosine_distances',
    'hamming_distances', 'binary_and', 'binary_or', 'binary_xor', 'popcount'
]

# GPU Matrix Operations

@compile_fn
def matmul(a: mx.array, b: mx.array) -> mx.array:
    """Matrix multiply via MLX (backend-optimized).

    When the kernel path is preferred (e.g., for large tiles), call the tiled
    GEMMs in `faissmlx/kernels/gemm_kernels.py`.
    """
    require_gpu("matmul")
    return mx.matmul(a, b)

@compile_fn
def l2_distances(x: mx.array, y: mx.array) -> mx.array:
    """L2 distances: (a-b)^2 = a^2 + b^2 - 2ab (full matrix)."""
    require_gpu("l2_distances")
    xx = mx.sum(mx.square(x), axis=1, keepdims=True)
    yy = mx.sum(mx.square(y), axis=1)
    xy = matmul(x, mx.transpose(y))
    xy2 = mx.add(xy, xy)
    return mx.subtract(mx.add(xx, yy), xy2)


def l2_distances_chunked(x: mx.array, y: mx.array, row_chunk: Optional[int] = None) -> mx.array:
    """L2 distances computed in row chunks to limit peak memory.

    Parameters
    - x: (n, d)
    - y: (m, d)
    - row_chunk: process this many rows of x per chunk; if None, chooses a
      heuristic based on device memory info (when available) or defaults.
    """
    n = int(x.shape[0]); d = int(x.shape[1]); m = int(y.shape[0])
    if row_chunk is None or row_chunk <= 0:
        # Heuristic: fit ~64MB worth of partial xy blocks (float32)
        try:
            import mlx.core.metal as metal
            info = metal.device_info()
            mem = int(info.get("max_recommended_working_set_size", 256 << 20))
        except Exception:
            mem = 256 << 20
        bytes_per_row = 4 * (m + d)  # xy row + x row square terms
        row_chunk = max(1, min(n, mem // max(bytes_per_row, 1)))

    outs = []
    for s in range(0, n, row_chunk):
        e = min(n, s + row_chunk)
        xs = x[s:e, :]
        xx = mx.sum(mx.square(xs), axis=1, keepdims=True)
        yy = mx.sum(mx.square(y), axis=1)
        xy = matmul(xs, mx.transpose(y))
        xy2 = mx.add(xy, xy)
        outs.append(mx.subtract(mx.add(xx, yy), xy2))
    return mx.concatenate(outs, axis=0)

@compile_fn
def cosine_distances(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized cosine distance computation.
    
    Args:
        x: First vectors (n, d)
        y: Second vectors (m, d)
        
    Returns:
        Distance matrix (n, m) on GPU
    """
    # Normalize and compute dot products efficiently
    require_gpu("cosine_distances")
    x_norm = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True))
    y_norm = mx.sqrt(mx.sum(mx.square(y), axis=1, keepdims=True))
    x = mx.divide(x, x_norm)
    y = mx.divide(y, y_norm)
    dot = matmul(x, mx.transpose(y))
    return mx.subtract(mx.ones_like(dot), dot)

# GPU Binary Operations

@compile_fn
def hamming_distances(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized Hamming distance computation.
    
    Args:
        x: First binary vectors (n, d) of uint8
        y: Second binary vectors (m, d) of uint8
        
    Returns:
        Distance matrix (n, m) of uint32 on GPU
    """
    require_gpu("hamming_distances")
    # Lookup table for Hamming weight (reuse from index.binary_index if available)
    try:
        from ..index.binary_index import HAMMING_TABLE as table  # type: ignore
    except Exception:
        table = mx.array([bin(i).count('1') for i in range(256)], dtype=mx.uint8)
    
    # Compute XOR then lookup Hamming weights
    # TODO: Add optimized implementation when MLX adds GPU-specific ops
    xor = mx.bitwise_xor(x[:, None, :], y[None, :, :])
    return mx.sum(table[xor], axis=2, dtype=mx.uint32)

@compile_fn
def binary_and(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized binary AND."""
    require_gpu("binary_and")
    return mx.bitwise_and(x, y)

@compile_fn
def binary_or(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized binary OR."""
    require_gpu("binary_or")
    return mx.bitwise_or(x, y)

@compile_fn
def binary_xor(x: mx.array, y: mx.array) -> mx.array:
    """GPU-optimized binary XOR."""
    require_gpu("binary_xor")
    return mx.bitwise_xor(x, y)

@compile_fn
def popcount(x: mx.array) -> mx.array:
    """GPU-optimized population count.
    
    Args:
        x: Input array of uint8
        
    Returns:
        Array with same shape containing bit counts
    """
    require_gpu("popcount")
    table = mx.array([bin(i).count('1') for i in range(256)], dtype=mx.uint8)
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

# Professional alias (neutral name)
class MemoryManager(GpuMemoryManager):
    pass
        
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
        size = 1
        for d in shape:
            size *= int(d)
        size *= int(element_size)
        
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
        size = 1
        for d in arr.shape:
            size *= int(d)
        size *= int(element_size)
        
        self.resources.free(size)
        # Actual freeing handled by MLX
