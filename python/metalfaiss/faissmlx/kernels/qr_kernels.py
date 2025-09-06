"""
Metal kernels (MLX JIT) for QR helpers

Kernels
- `qr_col_dot`: c = Qᵀ v (column‑parallel dot)
- `qr_update_vec`: v_out = v_in − Q c (row‑parallel update)

Contract and Design
- MLX `fast.metal_kernel` requires body‑only source; we place includes in `header`.
- Shapes are passed via a small `shape=[m,k]` buffer to avoid recompilation per call.
- Launch sizes are explicit; we pick multiples of the Apple execution width (32).

Optimization Notes
- `qr_col_dot` is memory‑bound; a simple loop over `m` per column is efficient
  for moderate sizes and pairs well with the MLX update path.
- `qr_update_vec` uses `fma` accumulation to reduce latency and improve math
  throughput; the kernel is 1D over rows.

References
- docs/mlx/Kernel-Guide.md:1
- docs/mlx/Comprehensive-MLX-Metal-Guide.md:1
- docs/metal/Shader-Optimization-Tips.md:102
"""

from typing import Tuple
import os
import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

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

_UPDATE_SRC = r"""
    uint gid = thread_position_in_grid.x;
    int m = int(shape[0]);
    int k = int(shape[1]);
    if (gid >= m) return;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        acc = fma(Q[gid * k + j], c[j], acc);
    }
    out[gid] = v[gid] - acc;
""";

_KERNEL_COL_DOT = mx.fast.metal_kernel(
    name="qr_col_dot",
    input_names=["Q", "v", "shape"],
    output_names=["out"],
    source=_COL_DOT_SRC,
    header=_HEADER,
    ensure_row_contiguous=True,
)

_KERNEL_UPDATE = mx.fast.metal_kernel(
    name="qr_update_vec",
    input_names=["Q", "c", "v", "shape"],
    output_names=["out"],
    source=_UPDATE_SRC,
    header=_HEADER,
    ensure_row_contiguous=True,
)


def project_coeffs(Q: mx.array, v: mx.array) -> mx.array:
    """Compute c = Qᵀ v using a simple, efficient Metal kernel.

    Parameters
    - `Q (m,k)`: columns (ideally) orthonormal
    - `v (m,)`

    Returns
    - `c (k,)`

    Notes
    - Launch is 1D over `k` with a modest threadgroup size (64). This keeps
      per‑thread work small and amortizes launch overhead.
    - Passing shape via buffer avoids recompilation across different `m,k`.
    """
    m, k = int(Q.shape[0]), int(Q.shape[1])
    shape = mx.array([m, k], dtype=mx.uint32)
    total = k
    tgroup = 64
    nthreads = ((total + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    threadgroup = (tgroup, 1, 1)

    (out,) = _KERNEL_COL_DOT(
        inputs=[Q, v, shape],
        output_shapes=[(k,)],
        output_dtypes=[Q.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return out


def update_vector(Q: mx.array, c: mx.array, v: mx.array) -> mx.array:
    """Compute v_out = v − Q c using a Metal kernel with `fma` accumulation.

    Parameters
    - `Q (m,k)`
    - `c (k,)`
    - `v (m,)`

    Returns
    - `v_out (m,)`

    Notes
    - Launch is 1D over `m` (rows). Each thread computes one output element and
      accumulates across `k` with `fma` for better throughput.
    - For large `k`, consider tiled updates (see GEMM kernels) if this becomes
      hot; for typical QR panel sizes, this form performs well.
    """
    m, k = int(Q.shape[0]), int(Q.shape[1])
    shape = mx.array([m, k], dtype=mx.uint32)
    total = m
    tgroup = 128
    nthreads = ((total + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    threadgroup = (tgroup, 1, 1)

    (out,) = _KERNEL_UPDATE(
        inputs=[Q, c, v, shape],
        output_shapes=[(m,)],
        output_dtypes=[Q.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return out
