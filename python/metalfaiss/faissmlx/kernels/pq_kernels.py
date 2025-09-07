"""
pq_kernels.py — PQ lookup‑table kernels for IVFADC (stubs)

Inspired by: Johnson et al., 2017 (GPU IVFADC with LUTs in shared memory)

Goals
- Build per‑query lookup tables (LUTs) in threadgroup memory and scan inverted
  lists with Asymmetric Distance Computation (ADC) while streaming results into
  a fast top‑k selector (see topk_kernels.py).
- Implement the Term 1/2/3 decomposition to minimize recomputation across τ
  probed lists when memory budget allows.

API targets
    build_pq_lut(q: mx.array, q1_centroid: mx.array, pq) -> mx.array
        Returns LUT of shape (M, ksub) in float32, suitable for threadgroup mem.
        - q: (d,) query; q1_centroid: (d,) coarse centroid; pq encodes centroids
          per subquantizer (M, ksub, dsub) or a compact equivalent.

    ivfpq_list_topk_adc(q: mx.array, codes: mx.array, ids: mx.array,
                        lut: mx.array, k: int) -> tuple[mx.array, mx.array]
        Compute ADC distances for one inverted list and return per‑list top‑k
        (k small, e.g., ≤ 32/64). Scans codes (L, M) uint8 and emits (k,).

    ivfpq_list_topk_adc_batch(Q: mx.array, Codes: mx.array, Ids: mx.array,
                               LUTs: mx.array, k: int) -> tuple[mx.array, mx.array]
        Batched variant (one threadgroup per query) for shared candidates.

Notes
- Threadgroup memory budget: b*ksub*4 bytes; b=M, typically 16–32; ksub=256.
  On Apple GPUs, TGM ≈ 32–64 KiB per TG, so M=16 (16 KiB) is safe.
"""

from __future__ import annotations
from typing import Tuple
import mlx.core as mx


def build_pq_lut(q: mx.array, q1_centroid: mx.array, pq) -> mx.array:
    """Build LUT (M, ksub) for ADC in threadgroup memory (stub).

    Term decomposition (Johnson et al., Eq. 11)
      - Term 1 (||q2(...)||^2): independent of query; precompute if memory allows
      - Term 2 (||x − q1(y)||^2): coarse distance; known per probed list
      - Term 3 (−2 ⟨x, q2(...)⟩): query dependent but invariant w.r.t list choice

    For now, the API reserves space to compute the combined LUT for the scanned
    list: LUT[m, code] = || q_m(code) − (q − q1_centroid)_m ||^2

    Returns
      - LUT: (M, ksub) float32 MLX array

    Status: stub — see PLAN.md → PQ kernels.
    """
    raise NotImplementedError("build_pq_lut: stub — see pq_kernels.py docstring")


def ivfpq_list_topk_adc(q: mx.array, codes: mx.array, ids: mx.array,
                        lut: mx.array, k: int) -> Tuple[mx.array, mx.array]:
    """Scan a single inverted list with ADC and return per‑list top‑k (stub).

    Parameters
      - q: (d,) query
      - codes: (L, M) uint8 PQ codes; ids: (L,) int32 vector ids
      - lut: (M, ksub) float32 LUT for this probed list
      - k: top‑k per list (small, e.g., ≤ 32/64)

    Returns
      - (vals, ids): (k,), (k,) MLX arrays ascending by distance

    Notes
      - Per‑thread local sums over M subquantizers; reduce into a simdgroup
        top‑k selector (see topk_kernels.simdgroup_topk_select) to produce the
        final (k) for the list.

    Status: stub — implemented later; see PLAN.md → PQ kernels / ADC scan.
    """
    raise NotImplementedError("ivfpq_list_topk_adc: stub — see PLAN.md → PQ kernels")


def ivfpq_list_topk_adc_batch(Q: mx.array, Codes: mx.array, Ids: mx.array,
                               LUTs: mx.array, k: int) -> Tuple[mx.array, mx.array]:
    """Batched ADC over shared candidates (stub).

    One threadgroup per query; reuses the same Codes/Ids with different LUTs.
    Returns (B, k) values and ids. Useful when many queries probe the same list.
    """
    raise NotImplementedError("ivfpq_list_topk_adc_batch: stub — batched ADC")

