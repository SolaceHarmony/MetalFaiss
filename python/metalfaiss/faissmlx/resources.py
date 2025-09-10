"""
resources.py — Neutral resource helpers (GPU-only project)

Provides professional, non-prefixed names that survive into production:
- Resources: tracks memory/limits; trivial wrapper until MLX exposes more
- MemoryManager: context manager for allocations; tracks byte counts

Compatibility aliases:
- GpuResources = Resources
- GpuMemoryManager = MemoryManager
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import mlx.core as mx

@dataclass
class Resources:
    device_id: int = 0
    max_memory: Optional[int] = None
    current_memory: int = 0

    def __post_init__(self):
        if self.device_id != 0:
            raise ValueError("Only device_id=0 is supported in GPU-only mode")
        if self.max_memory is not None and self.max_memory <= 0:
            raise ValueError("max_memory must be positive if provided")

    def allocate(self, size: int) -> bool:
        if self.max_memory is not None and self.current_memory + size > self.max_memory:
            return False
        self.current_memory += size
        return True

    def free(self, size: int) -> None:
        self.current_memory = max(0, self.current_memory - size)

    def reset(self) -> None:
        self.current_memory = 0


class MemoryManager:
    def __init__(self, resources: Optional[Resources] = None):
        self.resources = resources or Resources()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.resources.reset()

    def alloc(self, shape: Tuple[int, ...], dtype: str = "float32") -> mx.array:
        element_size = {
            "float32": 4, "float64": 8, "int32": 4, "int64": 8, "uint8": 1
        }.get(dtype, 4)
        size = 1
        for d in shape:
            size *= int(d)
        size *= int(element_size)
        if not self.resources.allocate(size):
            raise MemoryError("GPU memory limit exceeded")
        return mx.zeros(shape, dtype=dtype)

    def free(self, arr: mx.array) -> None:
        element_size = {
            mx.float32: 4, mx.float64: 8, mx.int32: 4, mx.int64: 8, mx.uint8: 1
        }.get(arr.dtype, 4)
        size = 1
        for d in arr.shape:
            size *= int(d)
        size *= int(element_size)
        self.resources.free(size)


# Compatibility aliases
GpuResources = Resources
GpuMemoryManager = MemoryManager

__all__ = [
    'Resources', 'MemoryManager', 'GpuResources', 'GpuMemoryManager'
]

