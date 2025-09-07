"""
topk_kernels.py — High‑performance top‑k selection kernels (stubs)

Inspired by: Johnson et al., "Billion-scale similarity search with GPUs" (2017)

Overview
- This module will house MLX + Metal kernels for fast top‑k selection, aligned
  to Apple GPUs and MLX’s `mx.fast.metal_kernel` contract. The primary target
  is a WarpSelect‑style design adapted to simdgroups, plus fused exact‑search
  (GEMM + top‑k) to avoid extra memory passes.

Status
- Stubs with comprehensive docstrings and API patterns are provided below.
  Each stub raises NotImplementedError until implemented.

Design notes (target behavior)
- simdgroup_topk_select(values, k): per‑simdgroup in‑register/threadgroup
  top‑k with odd‑merge networks. For Apple GPUs (simd width ≈ 32), use:
  - per‑thread small queue t (e.g., 2–8) in registers
  - a simdgroup shared queue of length k in threadgroup memory
  - conditionally merge via odd‑merge + bitonic networks using simdgroup_barrier
  - supports k ≤ 1024 (practically 32–256 sweet spot)
  - outputs both values and indices

- fused_gemm_topk(X, Y, k): exact brute‑force k‑NN via tiled dot product and
  in‑kernel top‑k to avoid materializing the full distance matrix. Decomposition
  D = ||x||^2 + ||y||^2 − 2 x·y. Options:
  - Use MLX GEMM (−2 x·y) then a tiled kernel that adds norms and selects top‑k
    reading tiles (2 passes total), or
  - Write a custom tiled GEMM that emits directly into the selector queues and
    only writes top‑k per query to global (1 pass over tiles).

API targets
    simdgroup_topk_select(values: mx.array, k: int) -> tuple[mx.array, mx.array]
        values: (n,) float32; returns (topk_vals, topk_idx) both (k,)
        Notes: public helper for small building blocks and tests; main usage is
               internal to fused kernels.

    fused_gemm_topk(X: mx.array, Y: mx.array, k: int,
                    distance: str = "l2") -> tuple[mx.array, mx.array]
        X: (q, d), Y: (n, d) float32; returns (vals, idx) shapes (q, k)
        distance = "l2" or "ip"; computes norms once per batch; tiles GEMM.

    fused_tile_topk(values_tile: mx.array, idx_base: int, k: int,
                    state: Optional[object]) -> object
        Implementation detail for streaming tiles through a persistent selector
        state. Allows composing with custom producers (e.g., PQ scans).

Integration plan
- Replace selection loops in ivf_kernels.py (and future pq_kernels.py) with
  simdgroup_topk_select for k≤32/64. For larger k, use two‑pass (per‑block top‑k
  then device‑merge) — we already have device_topk_merge.
- Add a brute‑force path: metalfaiss.faissmlx.brute.force_topk that calls
  fused_gemm_topk for exact search (baseline and comparator).
"""

from __future__ import annotations
from typing import Optional, Tuple
import mlx.core as mx


def simdgroup_topk_select(values: mx.array, k: int) -> Tuple[mx.array, mx.array]:
    """Top‑k via a simdgroup‑adapted WarpSelect (stub).

    Behavior
    - Maintains per‑thread small queues in registers and a shared simdgroup
      queue of length k in threadgroup memory.
    - Uses odd‑merge / bitonic networks to restore invariants when any lane’s
      head beats the shared queue tail.
    - Single pass over input; emits (topk_vals, topk_idx) sorted ascending.

    Constraints
    - Designed for Metal simdgroup width ≈ 32; k up to a few hundred.
    - values must be contiguous; if not, call `.reshape` or pass a view.

    Returns
    - (vals, idx): both (k,), MLX arrays on the current device.

    Status
    - Not implemented. See docs in this module and Johnson et al. (2017).
    """
    raise NotImplementedError("simdgroup_topk_select: stub — see PLAN.md → Top‑k kernels")


def fused_gemm_topk(X: mx.array, Y: mx.array, k: int, distance: str = "l2") -> Tuple[mx.array, mx.array]:
    """Exact brute‑force k‑NN via tiled GEMM + in‑kernel top‑k (stub).

    Parameters
    - X: (q, d) queries, Y: (n, d) database, float32
    - k: neighbors (k << n recommended)
    - distance: "l2" (default) or "ip" (inner product)

    Output
    - (vals, idx): (q, k), MLX arrays; vals sorted ascending for L2, descending
      for IP (by convention we can return negative sims to keep ascending).

    Design
    - Precompute ||X||^2 and ||Y||^2 once per batch.
    - Compute B = X @ Y^T (or −2 X @ Y^T for L2) in tiles; for each tile,
      stream its values through a persistent top‑k selector (per query) to avoid
      storing the whole distance matrix.
    - Minimize passes over the tile data and avoid extra global writes.

    Status
    - Not implemented. This function will orchestrate MLX matmul + Metal
      kernels (or fully custom kernels) to collect top‑k without materializing D.
    """
    raise NotImplementedError("fused_gemm_topk: stub — see PLAN.md → Brute‑force top‑k")


def fused_tile_topk(values_tile: mx.array, idx_base: int, k: int, state: Optional[object] = None) -> object:
    """Internal helper (stub): stream a values tile through a persistent top‑k state.

    Usage
    - Keep a per‑query selector state across tiles (threadgroup + per‑lane
      queues). Feed each tile and update the state; finalize to emit top‑k.

    Returns
    - Opaque state object to be passed to subsequent calls; a finalizer will
      extract the (vals, idx).

    Status
    - Stub for API illustration; exact type depends on Metal kernel backing.
    """
    raise NotImplementedError("fused_tile_topk: stub — building block for fused paths")

