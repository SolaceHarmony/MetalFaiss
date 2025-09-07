"""
hpc16x8.py — Extended-precision helpers for MLX (inspired by Ember ML)

This module provides a lightweight, MLX-only take on the limb/compensated
accumulation pattern for numerically sensitive QR/SVD computations when fp64
is not available on GPU. It keeps everything in MLX arrays (GPU by default)
and avoids host scalar pulls.

Initial surface:
- HPC16x8: container for high/low components; to_float32()
- kahan_sum(x): compensated summation for 1-D vectors (MLX-only)
- safe_norm2(x, eps): robust v^T v with clamp

Integration points (next steps):
- Use kahan_sum for projections and norms in QR when drift detected.
- Apply in SVD power-iteration accumulations as a guarded fallback.
"""
from __future__ import annotations
from typing import Optional, Tuple
import mlx.core as mx


def _add_comp(a: mx.array, b: mx.array) -> Tuple[mx.array, mx.array]:
    """TwoSum-like compensated add: returns (sum, err)."""
    s = mx.add(a, b)
    v = mx.subtract(s, a)
    e = mx.add(mx.subtract(a, mx.subtract(s, v)), mx.subtract(b, v))
    return s, e


class HPC16x8:
    """Minimal high/low container to hold extended-precision values.

    This is not a full limb implementation; it provides a convenient way to
    transport a compensated pair (high, low) and convert back to float32.
    """

    def __init__(self, high: mx.array, low: Optional[mx.array] = None):
        self.high = mx.array(high, dtype=mx.float32)
        self.low = mx.zeros_like(self.high) if low is None else mx.array(low, dtype=mx.float32)

    @classmethod
    def from_array(cls, x: mx.array) -> "HPC16x8":
        x32 = mx.array(x, dtype=mx.float32)
        low = mx.subtract(x, x32)
        return cls(x32, low)

    def to_float32(self) -> mx.array:
        s, e = _add_comp(self.high, self.low)
        return s  # drop residual for now; consumers can add e if desired


def kahan_sum(x: mx.array) -> mx.array:
    """Compensated sum for 1-D vectors (returns MLX scalar).

    Note: implemented as a simple loop over chunks to keep MLX graphs moderate.
    This is a compromise to stay MLX-only without host scalars. For long vectors
    consider banding/chunking before calling.
    """
    n = int(x.shape[0])
    if n == 0:
        return mx.array(0.0, dtype=mx.float32)
    s = mx.array(0.0, dtype=mx.float32)
    c = mx.array(0.0, dtype=mx.float32)
    # Elementwise Kahan; fine for modest sizes used in guard paths/tests.
    for i in range(n):
        xi = x[i]
        y = mx.subtract(xi, c)
        t = mx.add(s, y)
        c = mx.subtract(mx.subtract(t, s), y)
        s = t
    return s


def safe_norm2(x: mx.array, eps: float = 0.0) -> mx.array:
    """Robust squared norm v^T v with optional epsilon clamp (MLX scalar)."""
    v2 = mx.sum(mx.square(x))
    # Always apply clamp via MLX; eps may be 0 which is a no-op
    return mx.maximum(v2, mx.array(eps, dtype=x.dtype))