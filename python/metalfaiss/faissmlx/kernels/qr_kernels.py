"""
qr_kernels.py - Metal kernels (MLX JIT) for QR helpers

Provides a kernel to compute projection coefficients c = Q^T v
for Modified Gram–Schmidt, offloading dot-products to GPU threads.
"""

from typing import Tuple
import os
import mlx.core as mx

_COL_DOT_SRC = r"""
    uint gid = thread_position_in_grid.x;
    uint m = (uint)shape[0];
    uint k = (uint)shape[1];
    if (gid >= k) return;
    float acc = 0.0f;
    for (uint i = 0; i < m; ++i) {
        acc += Q[i * k + gid] * v[i];
    }
    out[gid] = acc;
""";


def _build_kernel():
    header = """#include <metal_stdlib>\nusing namespace metal;\n"""
    return mx.fast.metal_kernel(
        name="qr_col_dot",
        input_names=["Q", "v", "shape"],
        output_names=["out"],
        source=_COL_DOT_SRC,
        header=header,
        ensure_row_contiguous=True,
    )


_KERNEL = None


def project_coeffs(Q: mx.array, v: mx.array) -> mx.array:
    """Compute c = Q^T v using Metal kernel.

    Args:
        Q: (m, k) with orthonormal columns
        v: (m,)

    Returns:
        c: (k,)
    """
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    m, k = int(Q.shape[0]), int(Q.shape[1])
    shape = mx.array([m, k], dtype=mx.uint32)
    total = k
    tgroup = 64
    nthreads = ((total + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    threadgroup = (tgroup, 1, 1)

    (out,) = _KERNEL(
        inputs=[Q, v, shape],
        output_shapes=[(k,)],
        output_dtypes=[Q.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return out
