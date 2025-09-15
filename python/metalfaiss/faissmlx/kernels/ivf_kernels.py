"""
ivf_kernels.py - Metal kernels for IVF list scan + top-k selection

Implements a fused L2 distance + top-k kernel for one query against a set
of database vectors (typically the concatenation of selected IVF lists).

Notes
- Designed for small k (<= 32). Uses per-thread local top-k and a final
  threadgroup reduction to produce the final top-k distances and ids.
- Single-threadgroup kernel: threads stride across rows with step=tpb.
  This simplifies reduction. For very large lists, consider chunking and
  a two-pass merge on host or a second kernel.
"""

from __future__ import annotations
from typing import Tuple, Optional
import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_SRC_TOPK_L2 = r"""
    // Inputs:
    // Q: (d)
    // X: (m, d)
    // ids: (m)
    // shape = [m, d, k]
    const uint tpb = threads_per_threadgroup.x;
    const uint tid = thread_position_in_threadgroup.x;

    uint m = (uint)shape[0];
    uint d = (uint)shape[1];
    uint k = (uint)shape[2];

    // Per-thread local top-k (unsorted, track index of current worst)
    constexpr uint KMAX = 32u;
    thread float vals[KMAX];
    thread uint  idxs[KMAX];
    thread uint  count = 0u;
    thread uint  imax = 0u;    // index of current worst in vals[0..count)
    thread float vmax = -INFINITY; // track max among kept distances

    // Scan rows i = tid, tid+tpb, ...
    for (uint i = tid; i < m; i += tpb) {
        // L2 distance: sum_j (X[i,j] - Q[j])^2
        float acc = 0.0f;
        const device float* Xi = &X[i * d];
        for (uint j = 0; j < d; ++j) {
            float diff = Xi[j] - Q[j];
            acc = fma(diff, diff, acc);
        }

        uint kk = min(k, KMAX);
        if (kk == 0u) continue;
        if (count < kk) {
            vals[count] = acc;
            idxs[count] = i;
            // update vmax/imax
            if (acc > vmax || count == 0u) { vmax = acc; imax = count; }
            count++;
        } else {
            if (acc < vmax) {
                // replace worst
                vals[imax] = acc;
                idxs[imax] = i;
                // recompute worst among kk
                vmax = vals[0]; imax = 0u;
                for (uint t = 1u; t < kk; ++t) {
                    if (vals[t] > vmax) { vmax = vals[t]; imax = t; }
                }
            }
        }
    }

    // Threadgroup reduction: gather all local top-k and select final top-k
    threadgroup float tg_vals[1024]; // supports up to (tpb * min(k,32)) = 1024 when tpb=32 and k=32
    threadgroup uint  tg_idx [1024];

    uint kk = min(uint(shape[2]), KMAX);
    uint base = tid * kk;
    for (uint j = 0; j < kk; ++j) {
        tg_vals[base + j] = vals[j];
        tg_idx [base + j] = idxs[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        // Simple selection among all candidates (tpb * kk)
        uint total = tpb * kk;
        // Guard total if kk==0
        if (total == 0u) {
            // write +inf and zero ids
            for (uint j = 0; j < kk; ++j) { out_vals[j] = INFINITY; out_ids[j] = 0u; }
        } else {
            for (uint j = 0; j < kk; ++j) {
                // Find min in [j, total)
                uint best = j;
                float bestv = tg_vals[j];
                for (uint t = j + 1u; t < total; ++t) {
                    if (tg_vals[t] < bestv) { best = t; bestv = tg_vals[t]; }
                }
                // Swap into position j
                float tv = tg_vals[j]; tg_vals[j] = tg_vals[best]; tg_vals[best] = tv;
                uint  ti = tg_idx[j];  tg_idx[j]  = tg_idx[best];  tg_idx[best]  = ti;
            }
            // Write first k results (map list index to global id via ids array)
            for (uint j = 0; j < kk; ++j) {
                uint li = tg_idx[j];
                out_vals[j] = tg_vals[j];
                out_ids[j]  = ids[li];
            }
        }
    }
"""

_KERNEL_TOPK_L2 = mx.fast.metal_kernel(
    name="ivf_list_topk_l2",
    input_names=["Q", "X", "ids", "shape"],
    output_names=["out_vals", "out_ids"],
    header=_HEADER,
    source=_SRC_TOPK_L2,
    ensure_row_contiguous=True,
)


_SRC_TOPK_L2_BATCH = r"""
    // Batched variant: grid.x selects the query index qidx; one threadgroup per query.
    const uint tpb = threads_per_threadgroup.x;
    const uint tid = thread_position_in_threadgroup.x;

    uint m = (uint)shape[0];
    uint d = (uint)shape[1];
    uint k = (uint)shape[2];
    uint b = (uint)shape[3];

    uint qidx = threadgroup_position_in_grid.x; // 0..b-1
    if (qidx >= b) return;

    // Per-thread local top-k (unsorted, track index of current worst)
    constexpr uint KMAX = 32u;
    thread float vals[KMAX];
    thread uint  idxs[KMAX];
    thread uint  count = 0u;
    thread uint  imax = 0u;
    thread float vmax = -INFINITY;

    // Scan rows i = tid, tid+tpb, ...
    for (uint i = tid; i < m; i += tpb) {
        float acc = 0.0f;
        const device float* Xi = &X[i * d];
        const device float* Qq = &Q[qidx * d];
        for (uint j = 0; j < d; ++j) {
            float diff = Xi[j] - Qq[j];
            acc = fma(diff, diff, acc);
        }
        uint kk = min(k, KMAX);
        if (kk == 0u) continue;
        if (count < kk) {
            vals[count] = acc;
            idxs[count] = i;
            if (acc > vmax || count == 0u) { vmax = acc; imax = count; }
            count++;
        } else {
            if (acc < vmax) {
                vals[imax] = acc; idxs[imax] = i;
                vmax = vals[0]; imax = 0u;
                for (uint t = 1u; t < kk; ++t) { if (vals[t] > vmax) { vmax = vals[t]; imax = t; } }
            }
        }
    }

    // Threadgroup reduction into a temporary candidate array
    // Assume tpb*kk <= 1024
    constexpr uint MAXC = 1024u;
    threadgroup float tg_vals[MAXC];
    threadgroup uint  tg_idx [MAXC];
    uint kk = min(uint(shape[2]), KMAX);
    uint base = tid * kk;
    for (uint j = 0; j < kk; ++j) { tg_vals[base + j] = vals[j]; tg_idx[base + j] = idxs[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint total = tpb * kk;
        if (total == 0u) {
            for (uint j = 0; j < kk; ++j) { out_vals[qidx * kk + j] = INFINITY; out_ids[qidx * kk + j] = 0u; }
        } else {
            for (uint j = 0; j < kk; ++j) {
                uint best = j; float bestv = tg_vals[j];
                for (uint t = j + 1u; t < total; ++t) { if (tg_vals[t] < bestv) { best = t; bestv = tg_vals[t]; } }
                float tv = tg_vals[j]; tg_vals[j] = tg_vals[best]; tg_vals[best] = tv;
                uint  ti = tg_idx[j];  tg_idx[j]  = tg_idx[best];  tg_idx[best]  = ti;
            }
            for (uint j = 0; j < kk; ++j) {
                uint li = tg_idx[j];
                out_vals[qidx * kk + j] = tg_vals[j];
                out_ids[qidx * kk + j]  = ids[li];
            }
        }
    }
"""

_KERNEL_TOPK_L2_BATCH = mx.fast.metal_kernel(
    name="ivf_list_topk_l2_batch",
    input_names=["Q", "X", "ids", "shape"],
    output_names=["out_vals", "out_ids"],
    header=_HEADER,
    source=_SRC_TOPK_L2_BATCH,
    ensure_row_contiguous=True,
)


_SRC_TOPK_MERGE = r"""
    // Merge P partial top-k lists (vals_parts, ids_parts) into final top-k.
    // shape = [P, kk, k]
    const uint tid = thread_position_in_threadgroup.x;

    uint P  = (uint)shape[0];
    uint kk = (uint)shape[1];
    uint k  = (uint)shape[2];
    uint total = P * kk;

    // Copy inputs to threadgroup scratch since inputs are read-only
    threadgroup float tvals[2048]; // supports P*kk up to 2048
    threadgroup uint  tids [2048];
    for (uint t = tid; t < total && t < 2048u; t += threads_per_threadgroup.x) {
        tvals[t] = vals_parts[t];
        tids[t]  = ids_parts[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        // Simple selection on tvals/tids
        for (uint j = 0; j < k && j < total; ++j) {
            uint best = j; float bestv = tvals[j];
            for (uint t = j + 1u; t < total; ++t) { if (tvals[t] < bestv) { best = t; bestv = tvals[t]; } }
            float tv = tvals[j]; tvals[j] = tvals[best]; tvals[best] = tv;
            uint  ti = tids[j];  tids[j]  = tids[best];  tids[best]  = ti;
        }
        for (uint j = 0; j < k && j < total; ++j) { out_vals[j] = tvals[j]; out_ids[j] = tids[j]; }
        for (uint j = total; j < k; ++j) { out_vals[j] = INFINITY; out_ids[j] = 0u; }
    }
"""

_KERNEL_TOPK_MERGE = mx.fast.metal_kernel(
    name="ivf_topk_merge",
    input_names=["vals_parts", "ids_parts", "shape"],
    output_names=["out_vals", "out_ids"],
    header=_HEADER,
    source=_SRC_TOPK_MERGE,
    ensure_row_contiguous=True,
)

def ivf_list_topk_l2(Q: mx.array, X: mx.array, ids: mx.array, k: int, tpb: Optional[int] = None) -> Tuple[mx.array, mx.array]:
    """Compute top-k L2 distances and ids for one query Q against X.

    Args
    - Q: (d,) float32
    - X: (m,d) float32
    - ids: (m,) int32 (global ids corresponding to rows of X)
    - k: number of nearest neighbors (k <= 32)

    Returns
    - (vals, out_ids): both (k,), vals float32 ascending, ids int32
    """
    m, d = int(X.shape[0]), int(X.shape[1])
    kk = int(min(k, 32))
    shape = mx.array([m, d, kk], dtype=mx.uint32)
    # Single threadgroup; choose threads per threadgroup such that tpb*kk <= 1024
    if tpb is None:
        import os
        override = os.environ.get("METALFAISS_IVF_TPB")
        if override:
            try:
                tpb = int(override)
            except Exception:
                tpb = None
    if tpb is None:
        limit = max(1, 1024 // max(1, kk))
        # Round down to nearest multiple of 32, clamp to [32, 256]
        base = (limit // 32) * 32
        if base < 32:
            base = 32
        if base > 256:
            base = 256
        tpb = base
    grid = (1, 1, 1)
    threadgroup = (tpb, 1, 1)
    (vals, out_ids) = _KERNEL_TOPK_L2(
        inputs=[Q, X, ids, shape],
        output_shapes=[(kk,), (kk,)],
        output_dtypes=[Q.dtype, mx.int32],
        grid=grid,
        threadgroup=threadgroup,
    )
    return vals, out_ids


def ivf_list_topk_l2_chunked(Q: mx.array, X: mx.array, ids: mx.array, k: int, rows_per_chunk: int = 4096, tpb: Optional[int] = None) -> Tuple[mx.array, mx.array]:
    """Chunked variant: split rows and merge top-k on device.

    Calls ivf_list_topk_l2 per chunk, then merges partial (k) candidates
    entirely on device via `device_topk_merge` (k <= 32).
    """
    m = int(X.shape[0])
    kk = int(min(k, 32))
    parts_vals: list[mx.array] = []
    parts_ids: list[mx.array] = []

    for s in range(0, m, rows_per_chunk):
        e = min(m, s + rows_per_chunk)
        if e <= s:
            continue
        vals, oids = ivf_list_topk_l2(Q, X[s:e, :], ids[s:e], k, tpb=tpb)
        parts_vals.append(vals)
        parts_ids.append(oids)

    if not parts_vals:
        # Create +inf vector without Python floats: 1/0 → +inf (via mx.divide)
        infv = mx.divide(mx.ones((kk,), dtype=mx.float32), mx.zeros((kk,), dtype=mx.float32))
        return infv, mx.zeros((kk,), dtype=mx.int32)

    V = mx.stack(parts_vals)  # (P,kk)
    I = mx.stack(parts_ids)   # (P,kk)
    return device_topk_merge(V, I, k)


def ivf_list_topk_l2_batch(Q: mx.array, X: mx.array, ids: mx.array, k: int, tpb: Optional[int] = None) -> Tuple[mx.array, mx.array]:
    """Batched fused top-k: many queries (B,d) vs X (m,d).

    Returns
    - (vals, out_ids): both (B,k)
    """
    b, d = int(Q.shape[0]), int(Q.shape[1])
    m = int(X.shape[0])
    kk = int(min(k, 32))
    shape = mx.array([m, d, kk, b], dtype=mx.uint32)
    if tpb is None:
        import os
        override = os.environ.get("METALFAISS_IVF_TPB")
        if override:
            try:
                tpb = int(override)
            except Exception:
                tpb = None
    if tpb is None:
        limit = max(1, 1024 // max(1, kk))
        base = (limit // 32) * 32
        if base < 32:
            base = 32
        if base > 256:
            base = 256
        tpb = base
    grid = (b, 1, 1)
    threadgroup = (tpb, 1, 1)
    (vals, out_ids) = _KERNEL_TOPK_L2_BATCH(
        inputs=[Q, X, ids, shape],
        output_shapes=[(b, kk), (b, kk)],
        output_dtypes=[Q.dtype, mx.int32],
        grid=grid,
        threadgroup=threadgroup,
    )
    return vals, out_ids


def device_topk_merge(vals_parts: mx.array, ids_parts: mx.array, k: int) -> Tuple[mx.array, mx.array]:
    """Merge partial top-k (P,kk) on device into (k,).

    vals_parts, ids_parts: shape (P,kk)
    """
    P, kk = int(vals_parts.shape[0]), int(vals_parts.shape[1])
    shape = mx.array([P, kk, int(k)], dtype=mx.uint32)
    grid=(1,1,1); threadgroup=(32,1,1)
    (vals, ids) = _KERNEL_TOPK_MERGE(
        inputs=[vals_parts, ids_parts, shape],
        output_shapes=[(k,), (k,)],
        output_dtypes=[vals_parts.dtype, ids_parts.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return vals, ids


def ivf_list_topk_l2_chunked_device_merge(Q: mx.array, X: mx.array, ids: mx.array, k: int, rows_per_chunk: int = 4096, tpb: Optional[int] = None) -> Tuple[mx.array, mx.array]:
    """Chunked fused scan with device-side merge of partial top-k.

    Splits rows, collects per-chunk (k) candidates, then merges on device.
    """
    m = int(X.shape[0])
    kk = min(k, 32)
    parts_vals = []
    parts_ids = []
    for s in range(0, m, rows_per_chunk):
        e = min(m, s + rows_per_chunk)
        if e <= s:
            continue
        vals, oids = ivf_list_topk_l2(Q, X[s:e, :], ids[s:e], k, tpb=tpb)
        parts_vals.append(vals)
        parts_ids.append(oids)
    if not parts_vals:
        infv = mx.divide(mx.ones((kk,), dtype=mx.float32), mx.zeros((kk,), dtype=mx.float32))
        return infv, mx.zeros((kk,), dtype=mx.int32)
    V = mx.stack(parts_vals)  # (P,kk)
    I = mx.stack(parts_ids)   # (P,kk)
    return device_topk_merge(V, I, k)
