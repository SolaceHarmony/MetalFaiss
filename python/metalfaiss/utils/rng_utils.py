"""
rng_utils.py — Small helpers for MLX random key management.

MLX uses a splittable PRNG (JAX-style). Prefer explicit keys over global seed.
These helpers make it easy to create keys, split them, and draw random tensors
while threading keys deterministically through code and tests.
"""

from __future__ import annotations
import mlx.core as mx
from typing import Tuple, List

def new_key(seed: int) -> mx.array:
    """Create a new base PRNG key from a Python int seed."""
    return mx.random.key(int(seed))

def split2(key: mx.array) -> Tuple[mx.array, mx.array]:
    """Split a key into two independent subkeys."""
    return mx.random.split(key, num=2)

def split_n(key: mx.array, n: int) -> List[mx.array]:
    """Split a key into n independent subkeys."""
    ks = mx.random.split(key, num=int(n))
    # mx.random.split returns an array of keys; normalize to list
    return [ks[i] for i in range(int(ks.shape[0]))]

def normal(shape: tuple, *, key: mx.array, dtype=mx.float32) -> mx.array:
    """Draw a normal sample with explicit key."""
    return mx.random.normal(shape=shape, dtype=dtype, key=key)

def uniform(shape: tuple, *, key: mx.array, dtype=mx.float32, low=0.0, high=1.0) -> mx.array:
    """Draw a uniform sample with explicit key and optional bounds."""
    u = mx.random.uniform(shape=shape, dtype=dtype, key=key)
    if low == 0.0 and high == 1.0:
        return u
    span = mx.array(high - low, dtype=dtype)
    return mx.add(mx.multiply(u, span), mx.array(low, dtype=dtype))

