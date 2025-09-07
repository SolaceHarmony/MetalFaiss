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
            d2 = mx.sum((x[:, None, :] - cent[None, :, :]) ** 2, axis=2)
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
        d2 = mx.sum((x[:, None, :] - self._centroids[None, :, :]) ** 2, axis=2)
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
        d2 = mx.sum((x[:, None, :] - self._centroids[None, :, :]) ** 2, axis=2)
        labels = mx.argmin(d2, axis=1)
        residuals = x - self._centroids[labels]
        codes = self._pq.compute_codes(residuals)
        # Store per list
        for i in range(n):
            lid = int(labels[i].item())
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
            lut[m] = mx.sum(diff * diff, axis=1)
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
            d2 = mx.sum((self._centroids - q[None, :]) ** 2, axis=1)
            probe = mx.argsort(d2)[: self._nprobe].tolist()
            # ADC LUT for each list depends on q residual vs that list centroid
            cand_vals = []
            cand_ids = []
            for lid in probe:
                lid_int = int(lid)
                if not self._codes[lid_int]:
                    continue
                qres = q - self._centroids[lid_int]
                lut = self._adc_lut(qres)
                # Score each code in list
                for vid, code in self._codes[lid_int]:
                    # sum lut[m, code[m]] over m
                    s = 0.0
                    for m in range(self._pq.M):
                        s += float(lut[m, int(code[m].item())].item())
                    cand_vals.append(s)
                    cand_ids.append(vid)
            if not cand_vals:
                all_dists.append([float('inf')] * k)
                all_ids.append([0] * k)
                continue
            # Take top-k (smallest distances)
            import numpy as _np
            arr = _np.array(cand_vals)
            idx = _np.argpartition(arr, min(k, len(arr) - 1))[:k]
            sel = idx[_np.argsort(arr[idx])]
            all_dists.append([cand_vals[j] for j in sel])
            all_ids.append([cand_ids[j] for j in sel])
        return SearchResult(distances=all_dists, labels=all_ids)

    def reset(self) -> None:
        super().reset()
        self._codes = [[] for _ in range(self._nlist)]
        self._centroids = None
        self._quantizer.reset()

