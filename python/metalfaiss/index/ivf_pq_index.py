"""
ivf_pq_index.py - IVF index with Product Quantization (residual) using MLX

Implements a self-contained IVFPQ index:
- Trains coarse centroids via simple k-means (MLX)
- Trains a ProductQuantizer on training residuals
- Adds vectors by assigning to the nearest centroid and encoding residuals
- Searches by probing nprobe lists and computing ADC distances via LUTs

L2 metric only for now.
"""

from typing import List, Optional, Tuple
import mlx.core as mx

from .base_index import BaseIndex
from .flat_index import FlatIndex
from .product_quantizer import ProductQuantizer
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult


class IVFPQIndex(BaseIndex):
    """IVF index with residual Product Quantization (ADC search)."""

    def __init__(self, d: int, nlist: int, M: int, nbits: int = 8, metric_type: MetricType = MetricType.L2):
        super().__init__(d, metric=metric_type)
        if metric_type != MetricType.L2:
            raise ValueError("IVFPQIndex currently supports L2 metric only")
        self._nlist = nlist
        self._nprobe = 1
        self._centroids: Optional[mx.array] = None  # (nlist, d)
        self._pq = ProductQuantizer(d, M, nbits)
        # Inverted lists of (id, code) pairs
        self._codes: List[List[Tuple[int, mx.array]]] = [[] for _ in range(nlist)]
        # Coarse quantizer backed by centroid table for assignment
        self._quantizer = FlatIndex(d, metric_type=MetricType.L2)

    @property
    def nlist(self) -> int:
        return self._nlist

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value < 1:
            raise ValueError("nprobe must be >= 1")
        self._nprobe = value

    @property
    def pq(self) -> ProductQuantizer:
        return self._pq

    def _kmeans(self, x: mx.array, k: int, iters: int = 25) -> mx.array:
        """Simple L2 k-means returning centroids (k, d)."""
        n = int(x.shape[0])
        # init: sample k distinct points
        perm = mx.random.permutation(n)[:k]
        cent = x[perm]
        for _ in range(iters):
            d2 = mx.sum(mx.square(x[:, None, :] - cent[None, :, :]), axis=2)
            labels = mx.argmin(d2, axis=1)
            newc = []
            for j in range(k):
                mask = labels == j
                if mx.sum(mask) > 0:
                    newc.append(mx.mean(x[mask], axis=0))
                else:
                    newc.append(cent[j])
            cent = mx.stack(newc)
        return cent

    def train(self, xs: List[List[float]]) -> None:
        if not xs:
            raise ValueError("Empty training data")
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError("Dimension mismatch in training data")
        # Train coarse centroids
        self._centroids = self._kmeans(x, self._nlist)
        # Load quantizer with centroids
        self._quantizer.reset()
        self._quantizer.add(self._centroids.tolist())
        # Compute residuals for training PQ
        d2 = mx.sum(mx.square(x[:, None, :] - self._centroids[None, :, :]), axis=2)
        labels = mx.argmin(d2, axis=1)
        assigned = self._centroids[labels]
        residuals = x - assigned
        # Train PQ on residuals
        self._pq.train(residuals)
        self.is_trained = True

    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        if not self.is_trained:
            raise RuntimeError("Train index before adding vectors")
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError("Dimension mismatch in add()")
        n = int(x.shape[0])
        # Assign to nearest centroids
        d2 = mx.sum(mx.square(x[:, None, :] - self._centroids[None, :, :]), axis=2)
        labels = mx.argmin(d2, axis=1)
        residuals = x - self._centroids[labels]
        codes = self._pq.compute_codes(residuals)
        # Store per list
        for i in range(n):
            lid = int(labels[i])
            vid = ids[i] if ids is not None else (self.ntotal + i)
            self._codes[lid].append((int(vid), codes[i]))
        self.ntotal += n

    def _adc_lut(self, q: mx.array) -> mx.array:
        """Build LUT for query residual vs PQ centroids: (M, ksub)."""
        M = self._pq.M
        dsub = self._pq.dsub
        ksub = self._pq.ksub
        qsub = q.reshape(M, dsub)
        lut = mx.zeros((M, ksub), dtype=mx.float32)
        for m in range(M):
            diff = self._pq.centroids[m] - qsub[m][None, :]
            lut[m] = mx.sum(mx.square(diff), axis=1)
        return lut

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        if not self.is_trained:
            raise RuntimeError("Train index before search")
        xq = mx.array(xs, dtype=mx.float32)
        if xq.shape[1] != self.d:
            raise ValueError("Dimension mismatch in search()")
        nq = int(xq.shape[0])
        all_dists: List[List[float]] = []
        all_ids: List[List[int]] = []
        for i in range(nq):
            q = xq[i]
            # Coarse probe
        d2 = mx.sum(mx.square(self._centroids - q[None, :]), axis=1)
            probe = mx.argsort(d2)[: self._nprobe].tolist()
            # Accumulate candidates in MLX
            vals_accum: Optional[mx.array] = None
            ids_accum: Optional[mx.array] = None
            for lid in probe:
                lid_int = int(lid)
                codes_list = self._codes[lid_int]
                if not codes_list:
                    continue
                qres = q - self._centroids[lid_int]
                lut = self._adc_lut(qres)  # (M, ksub)
                # Stack codes and ids for this list
                ids_arr = mx.array([vid for (vid, _c) in codes_list], dtype=mx.int32)
                codes_mat = mx.stack([_c for (_vid, _c) in codes_list], axis=0)  # (L, M)
                L = int(codes_mat.shape[0])
                M = self._pq.M
                # Sum over subquantizers via gather
                scores = mx.zeros((L,), dtype=mx.float32)
                for m in range(M):
                    idx_m = codes_mat[:, m]
                    vals_m = lut[m][idx_m]
                    scores = scores + vals_m
                if vals_accum is None:
                    vals_accum = scores
                    ids_accum = ids_arr
                else:
                    vals_accum = mx.concatenate([vals_accum, scores], axis=0)
                    ids_accum = mx.concatenate([ids_accum, ids_arr], axis=0)
            if vals_accum is None or ids_accum is None or int(vals_accum.shape[0]) == 0:
                infv = mx.divide(mx.ones((k,), dtype=mx.float32), mx.zeros((k,), dtype=mx.float32))
                all_dists.append(infv)
                all_ids.append(mx.zeros((k,), dtype=mx.int32))
                continue
            # Top-k smallest via negative + topk
            kk = k if k <= int(vals_accum.shape[0]) else int(vals_accum.shape[0])
            neg = -vals_accum
            topv, topi = mx.topk(neg, kk, axis=0)
            sel_ids = ids_accum[topi]
            sel_vals = -topv
            all_dists.append(sel_vals)
            all_ids.append(sel_ids)
        return SearchResult(distances=mx.stack(all_dists), indices=mx.stack(all_ids))

    def reset(self) -> None:
        super().reset()
        self._codes = [[] for _ in range(self._nlist)]
        self._centroids = None
        self._quantizer.reset()
