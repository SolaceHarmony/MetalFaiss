"""
Tiled GEMM kernels (MLX + Metal)

Kernels
- `gemm_av`: B = A (m,n) × V (n,k) -> (m,k)
- `gemm_at_b`: Z = Aᵀ (n,m) × B (m,k) -> (n,k)

Design
- Body-only kernel sources with includes in `header` match the
  `mx.fast.metal_kernel` contract (no function signatures in `source`).
- Parameters (m, n, k) are passed in a small `shape` buffer to avoid
  recompilation across calls.
- 2D tiling with threadgroup shared memory enables coalesced loads and high
  arithmetic intensity; barriers synchronize phases.

Tile selection (hardware-aware)
- Tile sizes are chosen at import using `mlx.core.metal.device_info()` and can
  be overridden via env:
  - `METALFAISS_GEMM_TILE_AV="TMxT"` (e.g., `32x8`) – TN and TK are set to T.
  - `METALFAISS_GEMM_TILE_ATB="TNxTK"` (e.g., `8x32`).
  - Defaults: M3 → AV(32×8), AT_B(8×32); otherwise AV(16×16), AT_B(16×16).

Optimization Notes
- Coalesced loads: tiles are staged into `threadgroup` arrays and reused across
  inner-loop FMAs.
- fma accumulation: inner loops use `fma` to fuse multiply-add and exploit
  fast-math where available.
- Barrier scope: `threadgroup_barrier` is used since tiles are shared across
  multiple SIMD groups; use `simdgroup_barrier` only for warp-local exchanges.
- Avoid integer division/modulus in hot loops: kernels use 2D grid + tile math
  instead of computing indices via `/` and `%` on runtime values.

References
- docs/mlx/Kernel-Guide.md:120
- docs/mlx/Comprehensive-MLX-Metal-Guide.md:1
- docs/metal/Shader-Optimization-Tips.md:7
"""

from __future__ import annotations
from typing import Tuple
import os
import mlx.core as mx
try:
    import mlx.core.metal as metal
except Exception:  # pragma: no cover
    metal = None  # type: ignore
try:
    from .. import tuning as _tuning
except Exception:
    _tuning = None

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""


def _detect_device_name() -> str:
    try:
        if metal is None:
            return ""
        info = metal.device_info()
        return str(info.get("device_name", ""))
    except Exception:
        return ""


def _select_tile_av() -> Tuple[int, int]:
    """Select (TM, T) for AV kernel where TN=TK=T.

    Env override: METALFAISS_GEMM_TILE_AV="TMxT" (e.g., 32x8).
    Defaults: M3 → (32, 8); otherwise (16, 16).
    """
    env = os.environ.get("METALFAISS_GEMM_TILE_AV") or os.environ.get("METALFAISS_GEMM_TILE")
    if env:
        try:
            tm_s, t_s = env.lower().split("x")
            tm, t = int(tm_s), int(t_s)
            if tm * t <= 1024 and tm > 0 and t > 0:
                return tm, t
        except Exception:
            pass
    # Config file override
    if _tuning is not None:
        av_cfg, _ = _tuning.tiles_for_gemm()
        if av_cfg:
            try:
                tm_s, t_s = av_cfg.lower().split("x")
                tm, t = int(tm_s), int(t_s)
                if tm * t <= 1024 and tm > 0 and t > 0:
                    return tm, t
            except Exception:
                pass
    name = _detect_device_name().lower()
    if "m3" in name:
        return 32, 8
    return 16, 16


def _select_tile_atb() -> Tuple[int, int, int]:
    """Select (TN, TI, TK) for AT_B kernel.

    Env override: METALFAISS_GEMM_TILE_ATB="TNxTK"; TI fixed at 16.
    Defaults: M3 → (8, 16, 32); otherwise (16, 16, 16).
    """
    env = os.environ.get("METALFAISS_GEMM_TILE_ATB")
    if env:
        try:
            tn_s, tk_s = env.lower().split("x")
            tn, tk = int(tn_s), int(tk_s)
            if tn * tk <= 1024 and tn > 0 and tk > 0:
                return tn, 16, tk
        except Exception:
            pass
    # Config file override
    if _tuning is not None:
        _, atb_cfg = _tuning.tiles_for_gemm()
        if atb_cfg:
            try:
                tn_s, tk_s = atb_cfg.lower().split("x")
                tn, tk = int(tn_s), int(tk_s)
                if tn * tk <= 1024 and tn > 0 and tk > 0:
                    return tn, 16, tk
            except Exception:
                pass
    name = _detect_device_name().lower()
    if "m3" in name:
        return 8, 16, 32
    return 16, 16, 16

def _format_av_source(TM: int, T: int) -> str:
    """Return the AV kernel body with chosen tile sizes.

    Note: TN and TK are both set to T to keep local_x consistent for tile loads.
    """
    from string import Template
    tpl = Template(r"""
    // Threadgroup-tiled GEMM: C = A * B, here C=B, A=A, B=V
    // Shapes via shape buffer: [m, n, k]
    const uint TM = $TM; // tile size along m (rows of A / rows of C)
    const uint TN = $T;  // tile size along n (shared dimension)
    const uint TK = $T;  // tile size along k (cols of V / cols of C)

    threadgroup float Asub[TM][TN];
    threadgroup float Vsub[TN][TK];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint local_x = thread_position_in_threadgroup.x; // 0..TK-1 -> col in tile
    uint local_y = thread_position_in_threadgroup.y; // 0..TM-1 -> row in tile

    int block_x = int(threadgroup_position_in_grid.x); // tile col index
    int block_y = int(threadgroup_position_in_grid.y); // tile row index

    int row = block_y * int(TM) + int(local_y); // output row in [0,m)
    int col = block_x * int(TK) + int(local_x); // output col in [0,k)

    float acc = 0.0f;

    // Iterate over tiles of the shared dimension n
    int ntiles = (n + int(TN) - 1) / int(TN);
    for (int t = 0; t < ntiles; ++t) {
        // Global col in A / row in V tile
        int a_col = t * int(TN) + int(local_x);
        int v_row = t * int(TN) + int(local_y);

        // Load Asub[rowTile, p] and Vsub[p, colTile]
        float a_val = 0.0f;
        if (row < m && a_col < n) {
            a_val = A[row * n + a_col];
        }
        Asub[local_y][local_x] = a_val;

        float v_val = 0.0f;
        if (v_row < n && col < k) {
            v_val = V[v_row * k + col];
        }
        Vsub[local_y][local_x] = v_val;

        // Barrier required: tile data is shared across multiple SIMD groups
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over TN
        for (uint p = 0; p < TN; ++p) {
            acc = fma(Asub[local_y][p], Vsub[p][local_x], acc);
        }

        // Barrier required before loading the next tile iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < k) {
        C[row * k + col] = acc;
    }
""")
    return tpl.substitute(TM=TM, T=T)

def _format_at_b_source(TN: int, TI: int, TK: int) -> str:
    from string import Template
    tpl = Template(r"""
    // Threadgroup-tiled GEMM for Z = A^T * B
    // Shapes: A (m,n), B (m,k), Z (n,k), shape=[m,n,k]
    const uint TN = $TN; // tile size along n (rows of Z)
    const uint TI = $TI; // tile size along m (shared dimension)
    const uint TK = $TK; // tile size along k (cols of Z)

    threadgroup float Atile[TI][TN]; // A^T tile: [i in m][n in n]
    threadgroup float Btile[TI][TK];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint local_x = thread_position_in_threadgroup.x; // 0..TK-1 -> col in tile
    uint local_y = thread_position_in_threadgroup.y; // 0..TN-1 -> row in tile

    int block_x = int(threadgroup_position_in_grid.x); // tile col index in k
    int block_y = int(threadgroup_position_in_grid.y); // tile row index in n

    int rowN = block_y * int(TN) + int(local_y); // output row in n
    int colK = block_x * int(TK) + int(local_x); // output col in k

    float acc = 0.0f;

    int itiles = (m + int(TI) - 1) / int(TI); // walk shared dimension m
    for (int t = 0; t < itiles; ++t) {
        int i0 = t * int(TI);

        // Load Atile rows r assigned to threads striding by TN on local_y
        for (uint r = local_y; r < TI; r += TN) {
            int i = i0 + int(r);
            float a_val = 0.0f;
            if (i < m && rowN < n) {
                // A^T[rowN, i] = A[i, rowN]
                a_val = A[i * n + rowN];
            }
            Atile[r][local_y] = a_val; // Atile[TI][TN]
        }

        // Load Btile rows r assigned to threads striding by TK on local_x
        for (uint r = local_x; r < TI; r += TK) {
            int i = i0 + int(r);
            float b_val = 0.0f;
            if (i < m && colK < k) {
                b_val = B[i * k + colK];
            }
            Btile[r][local_x] = b_val; // Btile[TI][TK]
        }

        // Barrier required: tiles are consumed by multiple SIMD groups
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over TI
        for (uint p = 0; p < TI; ++p) {
            acc = fma(Atile[p][local_y], Btile[p][local_x], acc);
        }

        // Barrier before next iteration's loads
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (rowN < n && colK < k) {
        Z[rowN * k + colK] = acc;
    }
""")
    return tpl.substitute(TN=TN, TI=TI, TK=TK)

_KERNEL_AV = None
_KERNEL_AT_B = None
_TILES_AV: Tuple[int, int] | None = None
_TILES_ATB: Tuple[int, int, int] | None = None


def _build_av_kernel():
    """Create the tiled kernel for B = A × V.

    Implementation details
    - Inputs: `A (m,n)`, `V (n,k)`, `shape=[m,n,k]` (uint32)
    - Output: `C (m,k)`
    - Launch: 2D grid of (k/T, m/TM) with `threadgroup=(T, TM, 1)`
    - Uses threadgroup memory tiles and explicit `threadgroup_barrier` between
      load/accumulate phases.
    """
    global _TILES_AV
    if _TILES_AV is None:
        _TILES_AV = _select_tile_av()
    TM, T = _TILES_AV
    return mx.fast.metal_kernel(
        name="gemm_av_tiled",
        input_names=["A", "V", "shape"],
        output_names=["C"],
        header=_HEADER,
        source=_format_av_source(TM, T),
        ensure_row_contiguous=True,
    )


def _build_at_b_kernel():
    """Create the tiled kernel for Z = Aᵀ × B.

    Implementation details
    - Inputs: `A (m,n)`, `B (m,k)`, `shape=[m,n,k]`
    - Output: `Z (n,k)`
    - Launch: 2D grid of (k/TK, n/TN) with `threadgroup=(TK, TN, 1)`
    - Uses threadgroup memory tiles of Aᵀ and B along the shared m-dimension;
      explicit `threadgroup_barrier` between load/accumulate phases; `fma` in
      the inner loop.
    """
    global _TILES_ATB
    if _TILES_ATB is None:
        _TILES_ATB = _select_tile_atb()
    TN, TI, TK = _TILES_ATB
    return mx.fast.metal_kernel(
        name="gemm_at_b_tiled",
        input_names=["A", "B", "shape"],
        output_names=["Z"],
        header=_HEADER,
        source=_format_at_b_source(TN, TI, TK),
        ensure_row_contiguous=True,
    )


def set_gemm_tiles(av: str | Tuple[int, int] | None = None,
                   atb: str | Tuple[int, int] | None = None) -> None:
    """Override tile sizes and reset kernels.

    Args
    - av: "TMxT" or (TM, T) for AV kernel (TN=TK=T). If None, keep current.
    - atb: "TNxTK" or (TN, TK) for AT_B kernel (TI fixed at 16). If None, keep current.
    """
    global _TILES_AV, _TILES_ATB, _KERNEL_AV, _KERNEL_AT_B
    if av is not None:
        if isinstance(av, str) and "x" in av:
            tm_s, t_s = av.lower().split("x")
            _TILES_AV = (int(tm_s), int(t_s))
        elif isinstance(av, tuple):
            _TILES_AV = (int(av[0]), int(av[1]))
        _KERNEL_AV = None
    if atb is not None:
        if isinstance(atb, str) and "x" in atb:
            tn_s, tk_s = atb.lower().split("x")
            _TILES_ATB = (int(tn_s), 16, int(tk_s))
        elif isinstance(atb, tuple):
            _TILES_ATB = (int(atb[0]), 16, int(atb[1]))
        _KERNEL_AT_B = None


def get_gemm_tiles() -> Tuple[Tuple[int, int], Tuple[int, int, int]]:
    """Return the current tile sizes: (AV(TM,T), AT_B(TN,TI,TK))."""
    av = _TILES_AV or _select_tile_av()
    atb = _TILES_ATB or _select_tile_atb()
    return av, atb


def gemm_av(A: mx.array, V: mx.array) -> mx.array:
    """Compute B = A @ V with a shared‑memory tiled Metal kernel.

    Parameters
    - `A (m,n)`, row‑contiguous float32
    - `V (n,k)`, row‑contiguous float32

    Returns
    - `B (m,k)`

    Optimization
    - 2D tiles (16×16), coalesced loads, `threadgroup_barrier` between phases,
      `fma` accumulation in the inner loop. Avoids runtime integer division by
      using 2D grid math.

    See also
    - docs/mlx/Kernel-Guide.md:120
    - docs/metal/Shader-Optimization-Tips.md:148
    """
    use_kernel = os.environ.get("METALFAISS_USE_GEMM_KERNEL", "0") == "1"
    if not use_kernel:
        return mx.matmul(A, V)
    global _KERNEL_AV
    if _KERNEL_AV is None:
        _KERNEL_AV = _build_av_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(V.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    # Tile sizes (TM,T) -> threadgroup (T, TM, 1)
    TM, T = _TILES_AV or _select_tile_av()
    tiles_x = (k + T - 1) // T
    tiles_y = (m + TM - 1) // TM
    # MLX grid is threads, not groups
    grid = (tiles_x * T, tiles_y * TM, 1)
    threadgroup = (T, TM, 1)

    (B,) = _KERNEL_AV(
        inputs=[A, V, shape],
        output_shapes=[(m, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return B


def gemm_at_b(A: mx.array, B: mx.array) -> mx.array:
    """Compute Z = A.T @ B with a shared‑memory tiled Metal kernel.

    Parameters
    - `A (m,n)`, row‑contiguous float32
    - `B (m,k)`, row‑contiguous float32

    Returns
    - `Z (n,k)`

    Optimization
    - 2D tiles (16×16), explicit staging of Aᵀ and B tiles, `threadgroup_barrier`
      between phases, `fma` accumulation; avoids runtime `%` and `/` in hot loops.

    See also
    - docs/mlx/Kernel-Guide.md:120
    - docs/metal/Shader-Optimization-Tips.md:187
    """
    use_kernel = os.environ.get("METALFAISS_USE_GEMM_KERNEL", "0") == "1"
    if not use_kernel:
        return mx.matmul(mx.transpose(A), B)
    global _KERNEL_AT_B
    if _KERNEL_AT_B is None:
        _KERNEL_AT_B = _build_at_b_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(B.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    # Tile sizes (TN,TI,TK) -> threadgroup (TK, TN, 1)
    TN, TI, TK = _TILES_ATB or _select_tile_atb()
    tiles_x = (k + TK - 1) // TK
    tiles_y = (n + TN - 1) // TN
    # MLX grid is threads, not groups
    grid = (tiles_x * TK, tiles_y * TN, 1)
    threadgroup = (TK, TN, 1)

    (Z,) = _KERNEL_AT_B(
        inputs=[A, B, shape],
        output_shapes=[(n, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Z
