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
from typing import Tuple
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


def ivf_list_topk_l2(Q: mx.array, X: mx.array, ids: mx.array, k: int) -> Tuple[mx.array, mx.array]:
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
    # Single threadgroup; threads per group chosen modestly (32 or 64)
    tpb = 32
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
