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

Tile selection
- Square kernels use `METALFAISS_GEMM_TILE_SQ` (default 16).
- Non-square cooperative tile envs (`METALFAISS_GEMM_TILE_AV`, `METALFAISS_GEMM_TILE_ATB`)
  are only used by the experimental rectsafe AT_B path.

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
See also
- docs/mlx/GEMM-Kernels.md for flags, tuning, and validation tips.

Runtime toggles
- `METALFAISS_USE_GEMM_KERNEL=1` – enable Metal kernels (otherwise uses mx.matmul)
- `METALFAISS_GEMM_RECTSAFE=1` – use rectsafe AT_B mapping (unique-writer cooperative loads)
- `METALFAISS_GEMM_TILE_SQ=T` – square tile size for AV/AT_B square kernels (default 16)
- `METALFAISS_GEMM_DB=1` – enable double buffering (ping‑pong) for square kernels
- `METALFAISS_GEMM_V4=1` – attempt float4 vectorized loads when 16‑byte aligned; falls back otherwise
- `METALFAISS_GEMM_PAD_ATB=1` – pad AT_B shared tiles second dim (+1) to mitigate bank conflicts
"""

from __future__ import annotations
from typing import Tuple
import os
import json
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


# JSON configuration (optional)
_GEMM_CFG: dict | None = None


def _load_gemm_cfg() -> dict:
    """Load JSON config for GEMM kernels.

    Search order (first hit wins):
    1) Env path: METALFAISS_GEMM_CONFIG_JSON
    2) Package file: faissmlx/config/gemm_kernels.json (relative to this module)
    If none found or parse fails, return {}.
    """
    global _GEMM_CFG
    if _GEMM_CFG is not None:
        return _GEMM_CFG
    # 1) Env-specified path
    env_path = os.environ.get("METALFAISS_GEMM_CONFIG_JSON")
    candidates = []
    if env_path:
        candidates.append(env_path)
    # 2) Default packaged file
    here = os.path.dirname(__file__)  # .../faissmlx/kernels
    base = os.path.dirname(here)      # .../faissmlx
    default_path = os.path.join(base, "config", "gemm_kernels.json")
    candidates.append(default_path)
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                _GEMM_CFG = json.load(f) or {}
                return _GEMM_CFG
        except Exception:
            continue
    _GEMM_CFG = {}
    return _GEMM_CFG


def _get_cfg_bool(env: str, key: str, default: bool) -> bool:
    v = os.environ.get(env)
    if v is not None:
        v2 = v.strip().lower()
        return v2 in ("1", "true", "yes", "on")
    cfg = _load_gemm_cfg()
    if key in cfg:
        try:
            return bool(cfg[key])
        except Exception:
            pass
    return default


def _get_cfg_int(env: str, key: str, default: int) -> int:
    v = os.environ.get(env)
    if v is not None:
        try:
            return int(v)
        except Exception:
            return default
    cfg = _load_gemm_cfg()
    if key in cfg:
        try:
            return int(cfg[key])
        except Exception:
            pass
    return default


def _get_cfg_str(env: str, key: str) -> str | None:
    v = os.environ.get(env)
    if v:
        return v
    cfg = _load_gemm_cfg()
    val = cfg.get(key)
    return val if isinstance(val, str) and val else None


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
    env = _get_cfg_str("METALFAISS_GEMM_TILE_AV", "tile_av") or os.environ.get("METALFAISS_GEMM_TILE")
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
    env = _get_cfg_str("METALFAISS_GEMM_TILE_ATB", "tile_atb")
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


def _square_T_default() -> int:
    """Select the square tile T for the square‑tiled kernels.

    Env override: METALFAISS_GEMM_TILE_SQ (e.g., 8, 16, 32). Default 16.
    """
    t = _get_cfg_int("METALFAISS_GEMM_TILE_SQ", "square_T", 16)
    if t < 1:
        t = 1
    if t > 64:
        t = 64
    return t


def _use_db() -> bool:
    return _get_cfg_bool("METALFAISS_GEMM_DB", "double_buffer", False)


def _use_v4() -> bool:
    return _get_cfg_bool("METALFAISS_GEMM_V4", "vectorized_loads", False)


def _pad_atb() -> bool:
    return _get_cfg_bool("METALFAISS_GEMM_PAD_ATB", "pad_atb", False)


def _use_kernel_flag() -> bool:
    return _get_cfg_bool("METALFAISS_USE_GEMM_KERNEL", "use_gemm_kernel", False)


def _rectsafe_flag() -> bool:
    return _get_cfg_bool("METALFAISS_GEMM_RECTSAFE", "rectsafe", False)

# Removed legacy rectangular AV/AT_B formatters; square kernels are the production path.

def _format_at_b_source_rectsafe(TN: int, TI: int, TK: int) -> str:
    """Race-free loads for AT_B using unique writers per tile element.

    Each thread writes a unique (row r, col) element of Atile[TI][TN] or Btile[TI][TK]
    by striding r across TI using TN/TK respectively. Two barriers ensure
    visibility before use and before the next phase, preventing read/write hazards.
    """
    from string import Template
    tpl = Template(r"""
    // Threadgroup-tiled GEMM for Z = A^T * B (race-free loads)
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

        // Fill A^T tile: columns indexed by local_y (0..TN-1), rows cover [0..TI) in strides of TK
        for (uint r = local_x; r < TI; r += TK) {
            int i = i0 + int(r);
            float a_val = 0.0f;
            if (i < m && rowN < n) {
                a_val = A[i * n + rowN]; // A[i, rowN]
            }
            Atile[r][local_y] = a_val; // Atile[TI][TN]
        }

        // Fill B tile: columns indexed by local_x (0..TK-1), rows cover [0..TI) in strides of TN
        for (uint r = local_y; r < TI; r += TN) {
            int i = i0 + int(r);
            float b_val = 0.0f;
            if (i < m && colK < k) {
                b_val = B[i * k + colK]; // B[i, colK]
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

def _format_av_source_square_scalar(T: int, V4: bool) -> str:
    from string import Template
    tpl = Template(r"""
    // Square thread-tiled GEMM: C = A * V
    // Shapes via shape buffer: [m, n, k]
    const uint T = $T;

    // Pad second dimension to mitigate bank conflicts
    threadgroup float Asub[T][T+1];
    threadgroup float Vsub[T][T+1];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint tx = thread_position_in_threadgroup.x; // 0..T-1
    uint ty = thread_position_in_threadgroup.y; // 0..T-1

    int block_x = int(threadgroup_position_in_grid.x);
    int block_y = int(threadgroup_position_in_grid.y);
    int col = block_x * int(T) + int(tx);
    int row = block_y * int(T) + int(ty);

    float acc = 0.0f;
    int ntiles = (n + int(T) - 1) / int(T);
    for (int t = 0; t < ntiles; ++t) {
        int a_col = t * int(T) + int(tx);
        int v_row = t * int(T) + int(ty);

        // Vectorized group load: base lane = (tx & ~3)
        const bool USE_V4 = $V4;
        if (USE_V4) {
            uint base_tx = tx & 0xFFFFFFFCu; // multiple of 4 within tile
            int a_col_base = t * int(T) + int(base_tx);
            int col_base = block_x * int(T) + int(base_tx);
            uint idxA_base = uint(row) * uint(n) + uint(a_col_base);
            uint idxV_base = uint(v_row) * uint(k) + uint(col_base);
            bool in_tile = (base_tx + 3u) < T;
            bool a_ok = (row < m) && (a_col_base + 3 < n) && ((idxA_base & 3u) == 0u);
            bool v_ok = (v_row < n) && (col_base + 3 < k) && ((idxV_base & 3u) == 0u);
            bool group_ok = in_tile && a_ok && v_ok;
            if (group_ok) {
                if ((tx & 3u) == 0u) {
                    const device float4* A4 = (const device float4*)(A);
                    float4 a4 = A4[idxA_base >> 2];
                    Asub[ty][base_tx + 0] = a4.x;
                    Asub[ty][base_tx + 1] = a4.y;
                    Asub[ty][base_tx + 2] = a4.z;
                    Asub[ty][base_tx + 3] = a4.w;
                    const device float4* V4p = (const device float4*)(V);
                    float4 v4 = V4p[idxV_base >> 2];
                    Vsub[ty][base_tx + 0] = v4.x;
                    Vsub[ty][base_tx + 1] = v4.y;
                    Vsub[ty][base_tx + 2] = v4.z;
                    Vsub[ty][base_tx + 3] = v4.w;
                }
            } else {
                float a_val = 0.0f;
                if (row < m && a_col < n) { a_val = A[row * n + a_col]; }
                Asub[ty][tx] = a_val;
                float v_val = 0.0f;
                if (v_row < n && col < k) { v_val = V[v_row * k + col]; }
                Vsub[ty][tx] = v_val;
            }
        } else {
            float a_val = 0.0f;
            if (row < m && a_col < n) { a_val = A[row * n + a_col]; }
            Asub[ty][tx] = a_val;
            float v_val = 0.0f;
            if (v_row < n && col < k) { v_val = V[v_row * k + col]; }
            Vsub[ty][tx] = v_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < T; ++p) { acc = fma(Asub[ty][p], Vsub[p][tx], acc); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < k) { C[row * k + col] = acc; }
""")
    return tpl.substitute(T=T, V4=int(1 if V4 else 0))

def _format_av_source_square_db(T: int, V4: bool) -> str:
    from string import Template
    tpl = Template(r"""
    // Square thread-tiled GEMM with double buffering: C = A * V
    // Shapes via shape buffer: [m, n, k]
    const uint T = $T;

    threadgroup float As0[T][T+1];
    threadgroup float As1[T][T+1];
    threadgroup float Vs0[T][T+1];
    threadgroup float Vs1[T][T+1];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    int block_x = int(threadgroup_position_in_grid.x);
    int block_y = int(threadgroup_position_in_grid.y);
    int col = block_x * int(T) + int(tx);
    int row = block_y * int(T) + int(ty);

    float acc = 0.0f;
    int ntiles = (n + int(T) - 1) / int(T);
    if (ntiles == 0) { if (row < m && col < k) { C[row*k + col] = 0.0f; } return; }

    // Preload tile 0 into buffer 0
    {
        int v_row = 0 * int(T) + int(ty);
        const bool USE_V4 = $V4;
        if (USE_V4) {
            uint base_tx = tx & 0xFFFFFFFCu;
            int a_col_base = 0 * int(T) + int(base_tx);
            int col_base = block_x * int(T) + int(base_tx);
            uint idxA_base = uint(row) * uint(n) + uint(a_col_base);
            uint idxV_base = uint(v_row) * uint(k) + uint(col_base);
            bool in_tile = (base_tx + 3u) < T;
            bool a_ok = (row < m) && (a_col_base + 3 < n) && ((idxA_base & 3u) == 0u);
            bool v_ok = (v_row < n) && (col_base + 3 < k) && ((idxV_base & 3u) == 0u);
            bool group_ok = in_tile && a_ok && v_ok;
            if (group_ok) {
                if ((tx & 3u) == 0u) {
                    const device float4* A4 = (const device float4*)(A);
                    float4 a4 = A4[idxA_base >> 2];
                    As0[ty][base_tx + 0] = a4[0]; As0[ty][base_tx + 1] = a4[1]; As0[ty][base_tx + 2] = a4[2]; As0[ty][base_tx + 3] = a4[3];
                    const device float4* V4p = (const device float4*)(V);
                    float4 v4 = V4p[idxV_base >> 2];
                    Vs0[ty][base_tx + 0] = v4[0]; Vs0[ty][base_tx + 1] = v4[1]; Vs0[ty][base_tx + 2] = v4[2]; Vs0[ty][base_tx + 3] = v4[3];
                }
            } else {
                int a_col = 0 * int(T) + int(tx);
                float a_val = 0.0f; if (row < m && a_col < n) { a_val = A[row*n + a_col]; }
                As0[ty][tx] = a_val;
                float v_val = 0.0f; if (v_row < n && col < k) { v_val = V[v_row*k + col]; }
                Vs0[ty][tx] = v_val;
            }
        } else {
            int a_col = 0 * int(T) + int(tx);
            float a_val = 0.0f; if (row < m && a_col < n) { a_val = A[row*n + a_col]; }
            As0[ty][tx] = a_val;
            float v_val = 0.0f; if (v_row < n && col < k) { v_val = V[v_row*k + col]; }
            Vs0[ty][tx] = v_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop with ping-pong buffers
    for (int t = 0; t < ntiles - 1; ++t) {
        bool use0 = (t % 2) == 0;
        // Compute on current buffer
        if (use0) { for (uint p = 0; p < T; ++p) { acc = fma(As0[ty][p], Vs0[p][tx], acc); } }
        else      { for (uint p = 0; p < T; ++p) { acc = fma(As1[ty][p], Vs1[p][tx], acc); } }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Preload next tile (t+1) into the other buffer
        int v_row2 = (t + 1) * int(T) + int(ty);
        const bool USE_V4_2 = $V4;
        if (use0) {
            if (USE_V4_2) {
                uint base_tx = tx & 0xFFFFFFFCu;
                int a_col_base = (t + 1) * int(T) + int(base_tx);
                int col_base = block_x * int(T) + int(base_tx);
                uint idxA_base = uint(row) * uint(n) + uint(a_col_base);
                uint idxV_base = uint(v_row2) * uint(k) + uint(col_base);
                bool in_tile = (base_tx + 3u) < T;
                bool a_ok = (row < m) && (a_col_base + 3 < n) && ((idxA_base & 3u) == 0u);
                bool v_ok = (v_row2 < n) && (col_base + 3 < k) && ((idxV_base & 3u) == 0u);
                bool group_ok = in_tile && a_ok && v_ok;
                if (group_ok) {
                    if ((tx & 3u) == 0u) {
                        const device float4* A4 = (const device float4*)(A); float4 a4 = A4[idxA_base >> 2];
                        As1[ty][base_tx+0]=a4.x; As1[ty][base_tx+1]=a4.y; As1[ty][base_tx+2]=a4.z; As1[ty][base_tx+3]=a4.w;
                        const device float4* V4p = (const device float4*)(V); float4 v4 = V4p[idxV_base >> 2];
                        Vs1[ty][base_tx+0]=v4.x; Vs1[ty][base_tx+1]=v4.y; Vs1[ty][base_tx+2]=v4.z; Vs1[ty][base_tx+3]=v4.w;
                    }
                } else {
                    int a_col2 = (t + 1) * int(T) + int(tx);
                    float a_val = 0.0f; if (row < m && a_col2 < n) { a_val = A[row*n + a_col2]; } As1[ty][tx] = a_val;
                    float v_val = 0.0f; if (v_row2 < n && col < k) { v_val = V[v_row2*k + col]; } Vs1[ty][tx] = v_val;
                }
            } else {
                int a_col2 = (t + 1) * int(T) + int(tx);
                float a_val = 0.0f; if (row < m && a_col2 < n) { a_val = A[row*n + a_col2]; } As1[ty][tx] = a_val;
                float v_val = 0.0f; if (v_row2 < n && col < k) { v_val = V[v_row2*k + col]; } Vs1[ty][tx] = v_val;
            }
        } else {
            if (USE_V4_2) {
                uint base_tx = tx & 0xFFFFFFFCu;
                int a_col_base = (t + 1) * int(T) + int(base_tx);
                int col_base = block_x * int(T) + int(base_tx);
                uint idxA_base = uint(row) * uint(n) + uint(a_col_base);
                uint idxV_base = uint(v_row2) * uint(k) + uint(col_base);
                bool in_tile = (base_tx + 3u) < T;
                bool a_ok = (row < m) && (a_col_base + 3 < n) && ((idxA_base & 3u) == 0u);
                bool v_ok = (v_row2 < n) && (col_base + 3 < k) && ((idxV_base & 3u) == 0u);
                bool group_ok = in_tile && a_ok && v_ok;
                if (group_ok) {
                    if ((tx & 3u) == 0u) {
                        const device float4* A4 = (const device float4*)(A); float4 a4 = A4[idxA_base >> 2];
                        As0[ty][base_tx+0]=a4.x; As0[ty][base_tx+1]=a4.y; As0[ty][base_tx+2]=a4.z; As0[ty][base_tx+3]=a4.w;
                        const device float4* V4p = (const device float4*)(V); float4 v4 = V4p[idxV_base >> 2];
                        Vs0[ty][base_tx+0]=v4.x; Vs0[ty][base_tx+1]=v4.y; Vs0[ty][base_tx+2]=v4.z; Vs0[ty][base_tx+3]=v4.w;
                    }
                } else {
                    int a_col2 = (t + 1) * int(T) + int(tx);
                    float a_val = 0.0f; if (row < m && a_col2 < n) { a_val = A[row*n + a_col2]; } As0[ty][tx] = a_val;
                    float v_val = 0.0f; if (v_row2 < n && col < k) { v_val = V[v_row2*k + col]; } Vs0[ty][tx] = v_val;
                }
            } else {
                int a_col2 = (t + 1) * int(T) + int(tx);
                float a_val = 0.0f; if (row < m && a_col2 < n) { a_val = A[row*n + a_col2]; } As0[ty][tx] = a_val;
                float v_val = 0.0f; if (v_row2 < n && col < k) { v_val = V[v_row2*k + col]; } Vs0[ty][tx] = v_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final tile compute
    if ((ntiles - 1) >= 0) {
        bool use0 = ((ntiles - 1) % 2) == 0;
        if (use0) { for (uint p = 0; p < T; ++p) { acc = fma(As0[ty][p], Vs0[p][tx], acc); } }
        else      { for (uint p = 0; p < T; ++p) { acc = fma(As1[ty][p], Vs1[p][tx], acc); } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < k) { C[row * k + col] = acc; }
""")
    return tpl.substitute(T=T, V4=int(1 if V4 else 0))

def _format_at_b_source_square(T: int, PAD: bool, V4: bool) -> str:
    from string import Template
    tpl = Template(r"""
    // Threadgroup-tiled GEMM for Z = A^T * B (square tiles)
    // Shapes: A (m,n), B (m,k), Z (n,k), shape=[m,n,k]
    const uint T = $T;

    threadgroup float Atile[T][T+$PAD];
    threadgroup float Btile[T][T+$PAD];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    int block_x = int(threadgroup_position_in_grid.x);
    int block_y = int(threadgroup_position_in_grid.y);
    int colK = block_x * int(T) + int(tx);
    int rowN = block_y * int(T) + int(ty);

    float acc = 0.0f;
    int itiles = (m + int(T) - 1) / int(T);
    for (int t = 0; t < itiles; ++t) {
        int i0 = t * int(T);
        int ai = i0 + int(tx);
        float a_val = 0.0f;
        if (ai < m && rowN < n) { a_val = A[ai * n + rowN]; }
        Atile[ty][tx] = a_val;

        int bi = i0 + int(ty);
        // Optionally vectorized load for B across contiguous cols (k). Group leader writes for 4-wide group.
        const bool USE_V4 = $V4;
        if (USE_V4) {
            uint base_tx = tx & 0xFFFFFFFCu;
            int col_base = block_x * int(T) + int(base_tx);
            uint idxB_base = uint(bi) * uint(k) + uint(col_base);
            bool in_tile = (base_tx + 3u) < T;
            bool b_ok = (bi < m) && (col_base + 3 < k) && ((idxB_base & 3u) == 0u);
            bool group_ok = in_tile && b_ok;
            if (group_ok) {
                if ((tx & 3u) == 0u) {
                    const device float4* B4 = (const device float4*)(B);
                    float4 b4 = B4[idxB_base >> 2];
                    Btile[ty][base_tx + 0] = b4.x;
                    Btile[ty][base_tx + 1] = b4.y;
                    Btile[ty][base_tx + 2] = b4.z;
                    Btile[ty][base_tx + 3] = b4.w;
                }
            } else {
                float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; }
                Btile[ty][tx] = b_val;
            }
        } else {
            float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; }
            Btile[ty][tx] = b_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < T; ++p) { acc = fma(Atile[ty][p], Btile[p][tx], acc); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowN < n && colK < k) { Z[rowN * k + colK] = acc; }
""")
    return tpl.substitute(T=T, PAD=(1 if PAD else 0), V4=int(1 if V4 else 0))

def _format_at_b_source_square_db(T: int, PAD: bool, V4: bool) -> str:
    from string import Template
    tpl = Template(r"""
    // Threadgroup-tiled GEMM with double buffering: Z = A^T * B (square tiles)
    // Shapes: A (m,n), B (m,k), Z (n,k), shape=[m,n,k]
    const uint T = $T;

    threadgroup float A0[T][T+$PAD];
    threadgroup float A1[T][T+$PAD];
    threadgroup float B0[T][T+$PAD];
    threadgroup float B1[T][T+$PAD];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    int block_x = int(threadgroup_position_in_grid.x);
    int block_y = int(threadgroup_position_in_grid.y);
    int colK = block_x * int(T) + int(tx);
    int rowN = block_y * int(T) + int(ty);

    float acc = 0.0f;
    int itiles = (m + int(T) - 1) / int(T);
    if (itiles == 0) { if (rowN < n && colK < k) { Z[rowN*k + colK] = 0.0f; } return; }

    // Preload tile 0 into buffer 0
    {
        int i0 = 0;
        int ai = i0 + int(tx);
        float a_val = 0.0f; if (ai < m && rowN < n) { a_val = A[ai * n + rowN]; }
        A0[ty][tx] = a_val;

        int bi = i0 + int(ty);
        const bool USE_V4 = $V4;
        if (USE_V4) {
            uint base_tx = tx & 0xFFFFFFFCu;
            int col_base = block_x * int(T) + int(base_tx);
            uint idxB_base = uint(bi) * uint(k) + uint(col_base);
            bool in_tile = (base_tx + 3u) < T;
            bool b_ok = (bi < m) && (col_base + 3 < k) && ((idxB_base & 3u) == 0u);
            bool group_ok = in_tile && b_ok;
            if (group_ok) {
                if ((tx & 3u) == 0u) {
                    const device float4* B4 = (const device float4*)(B);
                    float4 b4 = B4[idxB_base >> 2];
                    B0[ty][base_tx+0]=b4.x; B0[ty][base_tx+1]=b4.y; B0[ty][base_tx+2]=b4.z; B0[ty][base_tx+3]=b4.w;
                }
            } else {
                float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; } B0[ty][tx] = b_val;
            }
        } else {
            float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; } B0[ty][tx] = b_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < itiles - 1; ++t) {
        bool use0 = (t % 2) == 0;
        if (use0) { for (uint p = 0; p < T; ++p) { acc = fma(A0[ty][p], B0[p][tx], acc); } }
        else      { for (uint p = 0; p < T; ++p) { acc = fma(A1[ty][p], B1[p][tx], acc); } }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int i0 = (t + 1) * int(T);
        int ai = i0 + int(tx);
        int bi = i0 + int(ty);
        if (use0) {
            float a_val = 0.0f; if (ai < m && rowN < n) { a_val = A[ai * n + rowN]; } A1[ty][tx] = a_val;
            const bool USE_V4_2 = $V4;
            if (USE_V4_2) {
                uint base_tx = tx & 0xFFFFFFFCu;
                int col_base = block_x * int(T) + int(base_tx);
                uint idxB_base = uint(bi) * uint(k) + uint(col_base);
                bool in_tile = (base_tx + 3u) < T;
                bool b_ok = (bi < m) && (col_base + 3 < k) && ((idxB_base & 3u) == 0u);
                bool group_ok = in_tile && b_ok;
            if (group_ok) {
                if ((tx & 3u) == 0u) { const device float4* B4 = (const device float4*)(B); float4 b4 = B4[idxB_base >> 2]; B1[ty][base_tx+0]=b4.x; B1[ty][base_tx+1]=b4.y; B1[ty][base_tx+2]=b4.z; B1[ty][base_tx+3]=b4.w; }
                } else { float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; } B1[ty][tx] = b_val; }
            } else {
                float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; } B1[ty][tx] = b_val;
            }
        } else {
            float a_val = 0.0f; if (ai < m && rowN < n) { a_val = A[ai * n + rowN]; } A0[ty][tx] = a_val;
            const bool USE_V4_2 = $V4;
            if (USE_V4_2) {
                uint base_tx = tx & 0xFFFFFFFCu;
                int col_base = block_x * int(T) + int(base_tx);
                uint idxB_base = uint(bi) * uint(k) + uint(col_base);
                bool in_tile = (base_tx + 3u) < T;
                bool b_ok = (bi < m) && (col_base + 3 < k) && ((idxB_base & 3u) == 0u);
                bool group_ok = in_tile && b_ok;
                if (group_ok) {
                    if ((tx & 3u) == 0u) { const device float4* B4 = (const device float4*)(B); float4 b4 = B4[idxB_base >> 2]; B0[ty][base_tx+0]=b4.x; B0[ty][base_tx+1]=b4.y; B0[ty][base_tx+2]=b4.z; B0[ty][base_tx+3]=b4.w; }
                } else { float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; } B0[ty][tx] = b_val; }
            } else {
                float b_val = 0.0f; if (bi < m && colK < k) { b_val = B[bi * k + colK]; } B0[ty][tx] = b_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((itiles - 1) >= 0) {
        bool use0 = ((itiles - 1) % 2) == 0;
        if (use0) { for (uint p = 0; p < T; ++p) { acc = fma(A0[ty][p], B0[p][tx], acc); } }
        else      { for (uint p = 0; p < T; ++p) { acc = fma(A1[ty][p], B1[p][tx], acc); } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (rowN < n && colK < k) { Z[rowN * k + colK] = acc; }
""")
    return tpl.substitute(T=T, PAD=(1 if PAD else 0), V4=int(1 if V4 else 0))

_KERNEL_AV = None
_KERNEL_AT_B = None
_TILES_AV: Tuple[int, int] | None = None
_TILES_ATB: Tuple[int, int, int] | None = None


def _build_av_kernel():
    """Create the tiled kernel for B = A × V.

    Implementation details
    - Inputs: `A (m,n)`, `V (n,k)`, `shape=[m,n,k]` (uint32)
    - Output: `C (m,k)`
    - Launch: grid launched in threads: `grid=(tiles_x*T, tiles_y*T, 1)`,
      `threadgroup=(T, T, 1)`
    - Uses threadgroup memory tiles and explicit `threadgroup_barrier` between
      load/accumulate phases.
    """
    global _TILES_AV
    if _TILES_AV is None:
        _TILES_AV = _select_tile_av()
    # Square tile size via env (default 16)
    T = _square_T_default()
    V4 = _use_v4()
    DB = _use_db()
    return mx.fast.metal_kernel(
        name="gemm_av_tiled",
        input_names=["A", "V", "shape"],
        output_names=["C"],
        header=_HEADER,
        source=(_format_av_source_square_db(T, V4) if DB else _format_av_source_square_scalar(T, V4)),
        ensure_row_contiguous=True,
    )


def _build_at_b_kernel():
    """Create the tiled kernel for Z = Aᵀ × B.

    Implementation details
    - Inputs: `A (m,n)`, `B (m,k)`, `shape=[m,n,k]`
    - Output: `Z (n,k)`
    - Launch: grid launched in threads: `grid=(tiles_x*T, tiles_y*T, 1)`,
      `threadgroup=(T, T, 1)`
    - Uses threadgroup memory tiles of Aᵀ and B along the shared m-dimension;
      explicit `threadgroup_barrier` between load/accumulate phases; `fma` in
      the inner loop.
    """
    global _TILES_ATB
    if _TILES_ATB is None:
        _TILES_ATB = _select_tile_atb()
    T = _square_T_default()
    PAD = _pad_atb()
    V4 = _use_v4()
    DB = _use_db()
    return mx.fast.metal_kernel(
        name="gemm_at_b_tiled",
        input_names=["A", "B", "shape"],
        output_names=["Z"],
        header=_HEADER,
        source=(_format_at_b_source_square_db(T, PAD, V4) if DB else _format_at_b_source_square(T, PAD, V4)),
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


def reset_gemm_kernels() -> None:
    """Reset cached GEMM kernels to rebuild with new env toggles.

    Use this when changing flags like METALFAISS_GEMM_DB, METALFAISS_GEMM_V4,
    METALFAISS_GEMM_PAD_ATB, or METALFAISS_GEMM_TILE_SQ between runs.
    """
    global _KERNEL_AV, _KERNEL_AT_B
    _KERNEL_AV = None
    _KERNEL_AT_B = None


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
    if not _use_kernel_flag():
        return mx.matmul(A, V)
    global _KERNEL_AV
    if _KERNEL_AV is None:
        _KERNEL_AV = _build_av_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(V.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    # Square tiles: threadgroup=(T,T), grid launched in threads
    T = _square_T_default()
    tiles_x = (k + T - 1) // T
    tiles_y = (m + T - 1) // T
    grid = (tiles_x * T, tiles_y * T, 1)
    threadgroup = (T, T, 1)

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
    if not _use_kernel_flag():
        return mx.matmul(mx.transpose(A), B)
    global _KERNEL_AT_B
    if _KERNEL_AT_B is None:
        _KERNEL_AT_B = _build_at_b_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(B.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    if _rectsafe_flag():
        # Build rectsafe kernel dynamically for race-free loads
        TN, TI, TK = _TILES_ATB or _select_tile_atb()
        ker = mx.fast.metal_kernel(
            name="gemm_at_b_rectsafe",
            input_names=["A", "B", "shape"],
            output_names=["Z"],
            header=_HEADER,
            source=_format_at_b_source_rectsafe(TN, TI, TK),
            ensure_row_contiguous=True,
        )
        tiles_x = (k + TK - 1) // TK
        tiles_y = (n + TN - 1) // TN
        grid = (tiles_x * TK, tiles_y * TN, 1)
        threadgroup = (TK, TN, 1)
        (Z,) = ker(
            inputs=[A, B, shape],
            output_shapes=[(n, k)],
            output_dtypes=[A.dtype],
            grid=grid,
            threadgroup=threadgroup,
        )
    else:
        # Square tiles: threadgroup=(T,T), grid launched in threads
        T = _square_T_default()
        tiles_x = (k + T - 1) // T
        tiles_y = (n + T - 1) // T
        grid = (tiles_x * T, tiles_y * T, 1)
        threadgroup = (T, T, 1)
        (Z,) = _KERNEL_AT_B(
            inputs=[A, B, shape],
            output_shapes=[(n, k)],
            output_dtypes=[A.dtype],
            grid=grid,
            threadgroup=threadgroup,
        )
    return Z
