"""
ivf_flat_index.py - IVF index with flat storage of vectors
"""

from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from .flat_index import FlatIndex
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult
from ..index.index_error import IndexError
from ..utils.sorting import topk_smallest_axis1
from ..distances import pairwise_L2sqr
import os
from ..faissmlx.flags import ivf_fused_enabled
try:
    from ..faissmlx.kernels.ivf_kernels import ivf_list_topk_l2
except Exception:
    ivf_list_topk_l2 = None  # type: ignore

class IVFFlatIndex(BaseIndex):
    """IVF index that stores raw vectors in inverted lists."""
    
    def __init__(self, quantizer: FlatIndex, d: int, nlist: int):
        """Initialize IVF flat index.
        
        Args:
            quantizer: Coarse quantizer (typically a FlatIndex)
            d: Vector dimension
            nlist: Number of inverted lists (partitions)
        """
        super().__init__(d)
        self._quantizer = quantizer
        self._nlist = nlist
        self._nprobe = 1  # Number of lists to probe during search
        self._invlists = [[] for _ in range(nlist)]
        
    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist
        
    @property
    def nprobe(self) -> int:
        """Number of lists to probe during search."""
        return self._nprobe
        
    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value < 1:
            raise ValueError("nprobe must be positive")
        self._nprobe = value
        
    @property
    def quantizer(self) -> FlatIndex:
        """The coarse quantizer used by this index."""
        return self._quantizer
        
    def train(self, xs: List[List[float]]) -> None:
        """Train the index.
        
        For IVFFlatIndex, this only trains the coarse quantizer.
        
        Args:
            xs: Training vectors
        """
        if not xs:
            raise ValueError("Empty training data")
            
        # Train quantizer (coarse centroids should be pre-built or provided)
        self._quantizer.train(xs)
        # Mark index as trained for add/search paths
        self._is_trained = True
        self.is_trained = True
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Assign vectors to lists using quantizer
        # `FlatIndex.search` returns (distances, indices); accept both tuple and SearchResult
        q_res = self._quantizer.search(xs, 1)
        if isinstance(q_res, tuple):  # (vals, idx)
            _, q_labels = q_res
        else:  # legacy SearchResult with `.labels`
            q_labels = q_res.labels  # type: ignore[attr-defined]

        # Append each vector to its assigned list
        n_new = int(x.shape[0])
        for i in range(n_new):
            li = int(q_labels[i, 0])
            vid = ids[i] if ids is not None else i
            self._invlists[li].append((vid, x[i]))
            
        self.ntotal += int(x.shape[0])
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Find nearest lists using quantizer
        coarse = self._quantizer.search(xs, self.nprobe)
        coarse_dists, coarse_labels = coarse if isinstance(coarse, tuple) else (coarse.distances, coarse.labels)  # type: ignore[attr-defined]
        
        # Search within selected lists
        out_vals: list[mx.array] = []
        out_ids: list[mx.array] = []
        for query, probe_labels in zip(x, coarse_labels):
            probe_vectors = []
            probe_ids = []

            # Gather vectors from selected lists
            for list_id in probe_labels:
                li = int(list_id)
                for vid, vec in self._invlists[li]:
                    probe_vectors.append(vec)
                    probe_ids.append(vid)

            if not probe_vectors:
                infv = mx.divide(mx.ones((k,), dtype=mx.float32), mx.zeros((k,), dtype=mx.float32))
                out_vals.append(infv)
                out_ids.append(mx.zeros((k,), dtype=mx.int32))
                continue

            probe_vectors = mx.stack(probe_vectors)
            ids_arr = mx.array(probe_ids, dtype=mx.int32)

            # Preferred: fused IVF top‑k kernel for L2
            if self.metric_type == MetricType.L2 and ivf_list_topk_l2 is not None:
                vals, oks = ivf_list_topk_l2(query, probe_vectors, ids_arr, k)
            else:
                # Fallback: compiled MLX pairwise + top-k (GPU‑only)
                drow = pairwise_L2sqr(query.reshape((1, -1)), probe_vectors)  # shape (1, m)
                vrow, irow = topk_smallest_axis1(drow, k)
                vals = vrow.reshape((-1,))
                oks = mx.take(ids_arr, irow.reshape((-1,)))
            out_vals.append(vals)
            out_ids.append(oks)

        D = mx.stack(out_vals) if out_vals else mx.zeros((len(x), k), dtype=mx.float32)
        I = mx.stack(out_ids) if out_ids else mx.zeros((len(x), k), dtype=mx.int32)
        return SearchResult(distances=D, indices=I)

    # Compatibility: expose `metric_type` like other index classes
    @property
    def metric_type(self) -> MetricType:
        return self.metric
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self._invlists = [[] for _ in range(self._nlist)]
        self._quantizer.reset()
