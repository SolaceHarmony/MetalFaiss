"""
Metal kernels (MLX JIT) for QR helpers

Kernels
- `qr_col_dot`: c = Qᵀ v (column‑parallel dot, simple per-thread loop)
- `qr_col_dot_simd`: c = Qᵀ v (one simdgroup per column; intra-warp reduction)
- `qr_update_vec`: v_out = v_in − Q c (row‑parallel update, fma)

Contract and Design
- MLX `fast.metal_kernel` requires body‑only source; we place includes in `header`.
- Shapes are passed via a small `shape=[m,k]` buffer to avoid recompilation per call.
- Launch sizes are explicit; we pick multiples of the Apple execution width (32).

Optimization Notes
- `qr_col_dot`: good baseline; one thread per column walks the rows.
- `qr_col_dot_simd`: each simdgroup (warp) accumulates one column using strided
  row walks and `simd_sum` to reduce within the warp; this improves throughput
  when `m` is large.
- `qr_update_vec` uses `fma` accumulation; kernel is 1D over rows.

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

# One simdgroup (warp) computes one column dot; stride by lane across rows
_COL_DOT_SIMD_SRC = r"""
    const uint WARP = 32u; // Apple warp width

    uint m = (uint)shape[0];
    uint k = (uint)shape[1];

    uint gid = thread_position_in_grid.x;     // global 1D thread index
    uint lane = (gid & (WARP - 1u));          // 0..31
    uint col = (gid / WARP);                  // one warp per column
    if (col >= k) return;

    float partial = 0.0f;
    for (uint i = lane; i < m; i += WARP) {
        partial = fma(Q[i * k + col], v[i], partial);
    }
    // Reduce within the warp via simd_sum; lane 0 writes the result
    float sum = simd_sum(partial);
    if (lane == 0u) {
        out[col] = sum;
    }
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

_KERNEL_COL_DOT_SIMD = mx.fast.metal_kernel(
    name="qr_col_dot_simd",
    input_names=["Q", "v", "shape"],
    output_names=["out"],
    source=_COL_DOT_SIMD_SRC,
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
    """Compute c = Qᵀ v using a simple or simdgroup-optimized Metal kernel.

    Parameters
    - `Q (m,k)`: columns (ideally) orthonormal
    - `v (m,)`

    Returns
    - `c (k,)`

    Notes
    - Auto-selects between a per-thread column loop and a simdgroup reduction
      based on a heuristic (large `m` favors simdgroup). Override via env:
        METALFAISS_QR_DOT=simple|simd
    - Passing shape via buffer avoids recompilation across different `m,k`.
    """
    m, k = int(Q.shape[0]), int(Q.shape[1])
    shape = mx.array([m, k], dtype=mx.uint32)
    mode = (os.environ.get("METALFAISS_QR_DOT") or "auto").lower()
    use_simd = False
    if mode == "simd":
        use_simd = True
    elif mode == "simple":
        use_simd = False
    else:
        # Heuristic: large m benefits from simdgroup reduction
        use_simd = m >= 512

    if use_simd:
        WARP = 32
        tg = (WARP, 1, 1)
        gx = k * WARP
        grid = (gx, 1, 1)
        (out,) = _KERNEL_COL_DOT_SIMD(
            inputs=[Q, v, shape],
            output_shapes=[(k,)],
            output_dtypes=[Q.dtype],
            grid=grid,
            threadgroup=tg,
        )
        return out
    else:
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
