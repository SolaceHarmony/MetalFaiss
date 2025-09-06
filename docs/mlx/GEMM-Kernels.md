# GEMM Kernels & Tuning (MLX + Metal)

This note documents the optimized tiled GEMM kernels used for the SVD Z‑step and other ops, how to enable them, and how to tune via runtime flags.

- Kernels: `gemm_av`: B = A (m,n) × V (n,k) → (m,k); `gemm_at_b`: Z = Aᵀ (n,m) × B (m,k) → (n,k)
- Source: `python/metalfaiss/faissmlx/kernels/gemm_kernels.py`

## Overview

- Square tiling: default `T=16`; grid launched in threads (`grid=(tiles_x*T, tiles_y*T)`, `threadgroup=(T,T)`)
- Shared tiles + barriers: two `threadgroup_barrier(mem_threadgroup)` per K‑tile iteration
- fma accumulation, row‑contiguous inputs, and coalesced cooperative loads
- Optional features: double buffering, vectorized loads, padding to mitigate bank conflicts

## Runtime Flags

- `METALFAISS_USE_GEMM_KERNEL=1`: enable Metal kernels (otherwise uses `mx.matmul`)
- `METALFAISS_GEMM_TILE_SQ`: square tile T for AV/AT_B square kernels (default `16`) — try 8/16/32
- `METALFAISS_GEMM_DB=1`: enable double buffering for square kernels (ping‑pong tiles)
- `METALFAISS_GEMM_V4=1`: attempt `float4` loads when aligned; safely falls back to scalar
- `METALFAISS_GEMM_PAD_ATB=1`: pad AT_B tiles’ second dimension (`[T][T+1]`) to reduce bank conflicts
- `METALFAISS_GEMM_RECTSAFE=1`: use the unique‑writer “rectsafe” AT_B mapping (experimental)

When changing flags between runs, call `reset_gemm_kernels()` to rebuild variants.

## Design Details

- Two‑phase per K‑tile: load tiles → barrier → FMA loop → barrier
- AV padding: `[T][T+1]` is always used to mitigate strided bank conflicts; AT_B padding is optional via flag
- Square tiles unify mapping and avoid runtime divisions/modulo hot‑path math

## Double Buffering (DB)

- Ping‑pong tile buffers: preload tile 0; then for t=0..ntiles−2 compute on buffer t%2 while loading tile t+1 into the other buffer; finally compute last tile
- Still exactly two `threadgroup_barrier(mem_threadgroup)` per tile iteration
- Benefit grows with larger tiles/shared work; neutral for small shapes

## Vectorized Loads (V4)

- Group leader (tx % 4 == 0) performs `float4` loads when 16‑byte aligned and in‑bounds; other threads in the 4‑wide group skip to avoid races
- AV: vectorize loads for both A (row‑contiguous) and V (row‑contiguous) when aligned
- AT_B: vectorize loads for B along k (contiguous) only; A is strided in n, kept scalar
- Guards ensure correctness: alignment, `T%4==0`, and bounds checks; fallback is scalar loads

## Padding (AT_B)

- `METALFAISS_GEMM_PAD_ATB=1` adds `+1` to the second dimension of AT_B tiles to reduce bank‑aligned strides during the inner p‑loop
- Low memory cost; often neutral to small positive gains

## Defaults & Tips

- Start with `T=16`; try `T=8` or `T=32` for specific shapes
- `V4=1` often helps; keep on unless shapes are highly irregular
- `DB=1`: more helpful for larger shared‑dim loops; may be neutral for small sizes
- `PAD_ATB=1`: safe default for T≥16; leave on when not sure

## Quick Start

- Enable kernels
  - `METALFAISS_USE_GEMM_KERNEL=1`
- Optional performance flags
  - `METALFAISS_GEMM_TILE_SQ=16 METALFAISS_GEMM_V4=1 METALFAISS_GEMM_PAD_ATB=1`
- Rebuild after toggle changes
  - `from metalfaiss.faissmlx.kernels import gemm_kernels as gk; gk.reset_gemm_kernels()`

## Validation

- Correctness quick check
  - `METALFAISS_USE_GEMM_KERNEL=1 python -m python.metalfaiss.unittest.test_gemm_flags_correctness`
- Bench overview
  - `METALFAISS_USE_GEMM_KERNEL=1 python -m python.metalfaiss.unittest.test_kernel_autotune_bench`
- Typical tolerances (float32): B max abs err < 5e−4; Z max abs err < 5e−3

## Troubleshooting

- Inputs must be row‑contiguous: builders enforce this; ensure upstream ops don’t create strided views
- Flag changes not taking effect: call `reset_gemm_kernels()` or restart process
- V4 not used: shape or index alignment may prevent vectorization (non‑multiple of 4); kernel falls back to scalar automatically

## API Notes

- `gemm_av(A, V) -> B` and `gemm_at_b(A, B) -> Z` accept MLX arrays; set `METALFAISS_USE_GEMM_KERNEL=1` to use kernels
- `set_gemm_tiles(av="TMxT", atb="TNxTK")` can override non‑square cooperative tiles (used by rectsafe or future variants)
- `get_gemm_tiles()` and `reset_gemm_kernels()` introspect/refresh configuration
