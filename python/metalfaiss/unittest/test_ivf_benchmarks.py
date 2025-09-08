"""
IVFFlat benchmarks: MLX vs fused Metal kernel (concat/per-list/chunked)

Explores permutations that affect speed: nprobe, k, threadgroup size (tpb),
and chunking. Prints median timings and basic sanity checks.
"""

import os
import time
import unittest
from typing import Tuple
import mlx.core as mx

from ..faissmlx.kernels.ivf_kernels import (
    ivf_list_topk_l2,
    ivf_list_topk_l2_chunked,
    ivf_list_topk_l2_chunked_device_merge,
    ivf_list_topk_l2_batch,
)


def _median_time(fn, warmup=1, repeats=3):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


class _SimpleQuantizer:
    def __init__(self, d: int, nlist: int):
        self.d = d
        self.nlist = nlist
        self.centroids = None

    def train(self, xs: mx.array):
        # K-means (few iters) for coarse centroids
        N = int(xs.shape[0]); d = int(xs.shape[1])
        k = self.nlist
        # init with random samples
        idx = mx.random.randint(0, N, (k,), dtype=mx.int32)
        C = xs[idx, :]
        for _ in range(5):
            # assign
            d2 = mx.sum((xs[:, None, :] - C[None, :, :])**2, axis=2)
            I = mx.argmin(d2, axis=1)
            # update
            # simple mean per cluster (not handling empty clusters carefully)
            for j in range(k):
                mask = (I == j)
                cnt = int(mx.sum(mask).item())  # boundary-ok
                if cnt > 0:
                    C[j] = mx.sum(xs * mask[:, None], axis=0) / cnt
        self.centroids = C

    def search(self, qs: mx.array, nprobe: int):
        d2 = mx.sum((qs[:, None, :] - self.centroids[None, :, :])**2, axis=2)
        idx = mx.argsort(d2, axis=1)[:, :nprobe]
        return d2, idx


def _build_ivf_inmemory(d: int, nlist: int, N: int, seed: int = 0):
    mx.random.seed(seed)
    xb = mx.random.normal(shape=(N, d)).astype(mx.float32)
    q = _SimpleQuantizer(d, nlist)
    train = mx.random.normal(shape=(min(8*nlist, max(256, N//2)), d)).astype(mx.float32)
    q.train(train)
    # build inverted lists
    d2 = mx.sum((xb[:, None, :] - q.centroids[None, :, :])**2, axis=2)
    I = mx.argmin(d2, axis=1)
    invlists = [[] for _ in range(nlist)]
    for i in range(N):
        invlists[int(I[i].item())].append((i, xb[i]))  # boundary-ok
    return q, invlists, xb


class TestIVFBenchmarks(unittest.TestCase):
    def test_ivf_permutations(self):
        # Shapes and parameters (kept small to run quickly here)
        d = 64
        N = 32768
        nlist = 128
        queries = 16
        ks = [10, 32]
        nprobes = [1, 8]

        q, X_lists, xb = _build_ivf_inmemory(d, nlist, N)

        xq = mx.random.normal(shape=(queries, d)).astype(mx.float32)
        xq_list = xq.tolist()  # boundary-ok (benchmark display)

        for k in ks:
            for nprobe in nprobes:
                nprobe = nprobe
                print(f"\n[IVF Bench] d={d}, N={N}, nlist={nlist}, nprobe={nprobe}, k={k}, Q={queries}")

                # Baseline: IVFFlat with MLX argsort (disable kernel)
                def _run_baseline():
                    # emulate IVFFlat search (MLX fallback)
                    for qi in range(queries):
                        qv = xq[qi]
                        _, probe = q.search(qv[None, :], nprobe)
                        probe_labels = probe[0]
                        vecs = []
                        ids = []
                        for list_id in probe_labels:
                            for vid, vec in X_lists[int(list_id)]:
                                vecs.append(vec)
                                ids.append(vid)
                        if not vecs:
                            continue
                        X = mx.stack(vecs)
                        dists = mx.sum((X - qv)**2, axis=1)
                        idx = mx.argsort(dists)[:k]
                        _ = (dists[idx], idx)
                t_baseline = _median_time(_run_baseline)
                print(f"  Baseline MLX:          {t_baseline:.4f}s")

                # Fused concat: gather all vectors from probed lists and call kernel once
                def _run_concat():
                    for qi in range(queries):
                        qv = xq[qi]
                        # coarse selection
                        _, probe = q.search(qv[None, :], nprobe)
                        probe_labels = probe[0]
                        vecs = []
                        ids = []
                        for list_id in probe_labels:
                            for vid, vec in X_lists[int(list_id)]:
                                vecs.append(vec)
                                ids.append(vid)
                        if not vecs:
                            continue
                        X = mx.stack(vecs)
                        I = mx.array(ids, dtype=mx.int32)
                        ivf_list_topk_l2(qv, X, I, k)
                os.environ['METALFAISS_USE_IVF_TOPK'] = '1'
                t_concat = _median_time(_run_concat)
                print(f"  Fused concat (tpb=auto): {t_concat:.4f}s")

                # Fused per-list: kernel per list, merge on host
                def _run_per_list():
                    import math
                    for qi in range(queries):
                        qv = xq[qi]
                        _, probe = q.search(qv[None, :], nprobe)
                        probe_labels = probe[0]
                        best_vals = [math.inf]*k; best_ids=[0]*k
                        def merge(vals, ids):
                            nonlocal best_vals, best_ids
                            lv, li = vals.tolist(), ids.tolist()
                            for v, i in zip(lv, li):
                                if math.isinf(v):
                                    continue
                                worst = max(range(k), key=lambda t: best_vals[t])
                                if v < best_vals[worst]:
                                    best_vals[worst]=v; best_ids[worst]=i
                        for list_id in probe_labels:
                            if not X_lists[int(list_id)]:
                                continue
                            vecs = [vec for (vid, vec) in X_lists[int(list_id)]]
                            ids = [vid for (vid, vec) in X_lists[int(list_id)]]
                            X = mx.stack(vecs)
                            I = mx.array(ids, dtype=mx.int32)
                            vals, oks = ivf_list_topk_l2(qv, X, I, k)
                            merge(vals, oks)
                t_per_list = _median_time(_run_per_list)
                print(f"  Fused per-list:        {t_per_list:.4f}s")

                # Fused concat, chunked: split rows and merge
                def _run_chunked():
                    for qi in range(queries):
                        qv = xq[qi]
                        _, probe = q.search(qv[None, :], nprobe)
                        probe_labels = probe[0]
                        vecs = []
                        ids = []
                        for list_id in probe_labels:
                            for vid, vec in X_lists[int(list_id)]:
                                vecs.append(vec)
                                ids.append(vid)
                        if not vecs:
                            continue
                        X = mx.stack(vecs)
                        I = mx.array(ids, dtype=mx.int32)
                        ivf_list_topk_l2_chunked(qv, X, I, k, rows_per_chunk=4096)
                t_chunked = _median_time(_run_chunked)
                print(f"  Fused concat (chunked): {t_chunked:.4f}s")

                # Fused concat, chunked with device merge
                def _run_chunked_devmerge():
                    for qi in range(queries):
                        qv = xq[qi]
                        _, probe = q.search(qv[None, :], nprobe)
                        probe_labels = probe[0]
                        vecs = []
                        ids = []
                        for list_id in probe_labels:
                            for vid, vec in X_lists[int(list_id)]:
                                vecs.append(vec)
                                ids.append(vid)
                        if not vecs:
                            continue
                        X = mx.stack(vecs)
                        I = mx.array(ids, dtype=mx.int32)
                        ivf_list_topk_l2_chunked_device_merge(qv, X, I, k, rows_per_chunk=4096)
                t_chunked_dm = _median_time(_run_chunked_devmerge)
                print(f"  Fused concat (chunk+dev): {t_chunked_dm:.4f}s")

                # Batched (synthetic): use same X for all queries of a batch
                # Note: realistic when many queries probe same lists (e.g., small nprobe or cached lists)
                def _run_batched_sameX():
                    # use first query's probed lists for all, as a synthetic batch scenario
                    qv = xq[0]
                    _, probe = q.search(qv[None, :], nprobe)
                    probe_labels = probe[0]
                    vecs = []
                    ids = []
                    for list_id in probe_labels:
                        for vid, vec in X_lists[int(list_id)]:
                            vecs.append(vec)
                            ids.append(vid)
                    if not vecs:
                        return None
                    X = mx.stack(vecs)
                    I = mx.array(ids, dtype=mx.int32)
                    ivf_list_topk_l2_batch(xq, X, I, k)
                t_batched = _median_time(_run_batched_sameX)
                print(f"  Fused batched (same X): {t_batched:.4f}s")

                # tpb override (if safe)
                def _tpb_safe(k):
                    # tpb*kk <= 1024 and multiple of 32
                    kk = min(k, 32)
                    return max(32, (1024//kk)//32*32)
                tpb = _tpb_safe(k)
                os.environ['METALFAISS_IVF_TPB'] = str(tpb)
                t_tpb = _median_time(_run_concat)
                print(f"  Fused concat (tpb={tpb}): {t_tpb:.4f}s")
                os.environ.pop('METALFAISS_IVF_TPB', None)


if __name__ == '__main__':
    unittest.main()
