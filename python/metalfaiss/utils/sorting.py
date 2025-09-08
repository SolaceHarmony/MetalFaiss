# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

import mlx.core as mx
from typing import Tuple, Optional

bucket_sort_verbose = 0

def fvec_argsort(vals: mx.array) -> mx.array:
    """Indirect sort of floating-point array.
    
    Args:
        vals: Array to sort
        
    Returns:
        Permutation array such that vals[perm[i + 1]] >= vals[perm[i]]
    """
    return mx.argsort(vals)

def fvec_argsort_parallel(vals: mx.array) -> mx.array:
    """Parallel indirect sort of floating-point array.
    
    Args:
        vals: Array to sort
        
    Returns:
        Permutation array such that vals[perm[i + 1]] >= vals[perm[i]]
    """
    # MLX handles parallelization internally
    return fvec_argsort(vals)

"""
Legacy CPU-only bucket sorts and hash tables removed for pure-MLX build.
If needed for CPU utilities, reintroduce as a separate optional module.
"""

# --- MLX helpers -----------------------------------------------------------

def mlx_topk(
    x: mx.array,
    k: int,
    *,
    axis: int = -1,
    largest: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Return (values, indices) of top-k elements along an axis using pure MLX ops.

    MLX's `mx.topk` currently returns values only; many call sites expect both
    values and indices. This helper provides a consistent interface using
    `mx.argsort` + `mx.take`, avoiding any Python/NumPy math on MLX arrays.

    Args:
        x: Input array.
        k: Number of elements to select (clip to axis length).
        axis: Axis along which to select.
        largest: If True, select largest k; otherwise select smallest k.

    Returns:
        Tuple of (values, indices), both MLX arrays with the same shape.
    """
    # Normalize axis and clip k
    ndim = len(x.shape)
    ax = axis if axis >= 0 else (ndim + axis)
    if ax < 0 or ax >= ndim:
        raise ValueError(f"axis out of range: {axis}")
    axis_len = int(x.shape[ax])
    kk = k if k <= axis_len else axis_len

    # Order along axis via argsort (ascending). For largest, sort -x via mx.negative.
    sort_source = mx.negative(x) if largest else x
    order = mx.argsort(sort_source, axis=ax)

    # Slice first k indices along the axis.
    sl = [slice(None)] * ndim
    sl[ax] = slice(0, kk)
    top_idx = order[tuple(sl)]

    # Gather values along axis using the selected indices.
    # Gather values along axis using the selected indices.
    if ndim == 2 and ax == 1:
        n, m = int(x.shape[0]), int(x.shape[1])
        rows = mx.arange(n, dtype=top_idx.dtype).reshape((n, 1))
        flat_idx = mx.add(mx.multiply(rows, mx.array(m, dtype=top_idx.dtype)), top_idx)
        top_vals = mx.take(x.reshape((n * m,)), flat_idx)
    else:
        top_vals = mx.take(x, top_idx, axis=ax)
    return top_vals, top_idx
