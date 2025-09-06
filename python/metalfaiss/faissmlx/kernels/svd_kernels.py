"""
SVD subspace power-iteration kernels (baseline)

This module provides a simple, correct baseline kernel to compute:
    Z = Aᵀ (A V)
for a block of vectors V. It is intentionally naive (nested loops) and
serves as a contrast to the tiled two-kernel approach in `gemm_kernels.py`.

Notes
- The baseline uses a 1D launch over `n*k` and maps indices via `/` and `%` by
  a runtime `k`. This is pedagogical; the optimized path avoids such divides by
  using a 2D grid and shared-memory tiles.
- For real workloads, prefer the tiled approach (A@V then Aᵀ@B) exposed in
  `python/metalfaiss/faissmlx/kernels/gemm_kernels.py`.

Usage
    from .svd_kernels import power_iter_step
    Z = power_iter_step(A, V)  # A (m,n), V (n,k), Z (n,k)
"""

from typing import Tuple
import mlx.core as mx

# Keep includes in header; body is function-less source
_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_BODY_AT_A_V = r"""
    // Inputs: A (m,n), V (n,k), shape = [m, n, k]
    // Output: Z (n,k) = A^T (A V)
    uint gid = thread_position_in_grid.x;
    uint m = (uint)shape[0];
    uint n = (uint)shape[1];
    uint k = (uint)shape[2];
    uint total = n * k;
    if (gid >= total) return;

    uint col = gid % k;     // 0..k-1
    uint rowN = gid / k;    // 0..n-1 (row index in Z / A^T)

    float acc = 0.0f;
    // Compute Z[rowN, col] = sum_i A[i,rowN] * (AV)[i,col]
    for (uint i = 0; i < m; ++i) {
        float a_i_rowN = A[i * n + rowN];
        // (AV)[i,col] = sum_j A[i,j] * V[j,col]
        float av = 0.0f;
        for (uint j = 0; j < n; ++j) {
            av += A[i * n + j] * V[j * k + col];
        }
        acc += a_i_rowN * av;
    }
    Z[rowN * k + col] = acc;
"""

_KERNEL_AT_A_V = None


def _build_at_a_v_kernel():
    """Create the baseline kernel Z = Aᵀ (A V).

    Implementation details
    - Inputs: `A (m,n)`, `V (n,k)`, `shape=[m,n,k]`
    - Output: `Z (n,k)`
    - Launch: 1D over `n*k`. Uses `/` and `%` to derive (row,col) indices.
      This is slower than a 2D grid and kept for demonstration.
    """
    return mx.fast.metal_kernel(
        name="svd_at_a_v",
        input_names=["A", "V", "shape"],
        output_names=["Z"],
        header=_HEADER,
        source=_BODY_AT_A_V,
        ensure_row_contiguous=True,
    )


def power_iter_step(A: mx.array, V: mx.array) -> mx.array:
    """Compute one Z-step for block power iteration: Z = Aᵀ (A V).

    Parameters
    - `A (m,n)`
    - `V (n,k)`

    Returns
    - `Z (n,k)`

    Notes
    - This baseline kernel is mostly for correctness and teaching. For speed,
      use the tiled kernels in `gemm_kernels.py` via `svd.topk_svd(..., use_kernel=True)`.
    """
    global _KERNEL_AT_A_V
    if _KERNEL_AT_A_V is None:
        _KERNEL_AT_A_V = _build_at_a_v_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(V.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    total = n * k
    tgroup = 256
    nthreads = ((total + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    threadgroup = (tgroup, 1, 1)

    (Z,) = _KERNEL_AT_A_V(
        inputs=[A, V, shape],
        output_shapes=[(n, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Z
