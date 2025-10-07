from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from ..utils.search_result import SearchResult
from ..types.metric_type import MetricType
from ..distances import (
    pairwise_L2sqr,
    pairwise_L1,
    pairwise_Linf,
    pairwise_extra_distances,
    pairwise_jaccard,
)
from ..utils.sorting import topk_smallest_axis1

class FlatIndex(BaseIndex):
    """Flat index implementation using MLX.
    
    This is the most basic index type that stores vectors in memory
    and performs exact nearest neighbor search.
    """
    
    def __init__(self, d: int, metric_type: MetricType = MetricType.L2):
        """Initialize flat index.
        
        Args:
            d: Dimension of vectors to index
            metric_type: Distance metric to use
        """
        super().__init__(d, metric=metric_type)
        self.metric_type = metric_type
        self._vectors: Optional[mx.array] = None
        self._ids: Optional[mx.array] = None
        self._id_to_row: dict[int, int] = {}
        self.is_trained = True
        
    @classmethod
    def from_index(cls, index: BaseIndex) -> Optional['FlatIndex']:
        """Create FlatIndex from generic index if possible.
        
        Args:
            index: Index to convert
            
        Returns:
            FlatIndex if conversion possible, None otherwise
        """
        if isinstance(index, cls):
            return index
        return None
        
    def train(self, xs: List[List[float]]) -> None:
        """Flat indexes are always considered trained."""
        self.is_trained = True
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs (ignored in flat index)
        """
        if ids is not None and len(ids) != len(xs):
            raise ValueError("Length of ids must match number of vectors")

        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")

        start = self.ntotal
        n_new = int(x.shape[0])
        if ids is not None:
            id_arr = mx.array(ids, dtype=mx.int64)
            for offset, val in enumerate(ids):
                self._id_to_row[int(val)] = start + offset
        else:
            id_arr = mx.arange(start, start + n_new, dtype=mx.int64)
            for offset in range(n_new):
                self._id_to_row[start + offset] = start + offset

        if self._vectors is None:
            self._vectors = x
            self._ids = id_arr
        else:
            self._vectors = mx.concatenate([self._vectors, x], axis=0)
            self._ids = mx.concatenate([self._ids, id_arr], axis=0)  # type: ignore[arg-type]

        self.ntotal = int(self._vectors.shape[0])
        mx.eval(self._vectors)
        mx.eval(self._ids)
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        k = min(k, self.ntotal)
        
        # Compute distances matrix (lower is better), except for INNER_PRODUCT
        mt = self.metric_type
        if mt == MetricType.L2:
            distances = pairwise_L2sqr(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.L1:
            distances = pairwise_L1(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.Linf:
            distances = pairwise_Linf(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.Canberra:
            distances = pairwise_extra_distances(x, self._vectors, "Canberra")
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.BrayCurtis:
            distances = pairwise_extra_distances(x, self._vectors, "BrayCurtis")
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.JensenShannon:
            distances = pairwise_extra_distances(x, self._vectors, "JensenShannon")
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.Jaccard:
            distances = pairwise_jaccard(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.INNER_PRODUCT:
            sims = mx.matmul(x, self._vectors.T)
            distances = mx.negative(sims)
            vals, idx = topk_smallest_axis1(distances, k)
        else:
            distances = pairwise_L2sqr(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)

        selected_ids = mx.take(self._ids, idx, axis=0)
        return SearchResult(distances=vals, indices=selected_ids)
        
    def xb(self) -> List[List[float]]:
        """Get stored vectors.
        
        Returns:
            Stored vectors as an MLX array
        """
        if self._vectors is None:
            return mx.zeros((0, self.d), dtype=mx.float32)
        return self._vectors

    def reconstruct(self, key: int) -> mx.array:
        """Reconstruct the stored vector corresponding to the given ID."""
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
        if key not in self._id_to_row:
            raise ValueError(f"Unknown key {key}")
        return self._vectors[self._id_to_row[key]]

    def reset(self) -> None:
        super().reset()
        self._vectors = None
        self._ids = None
        self._id_to_row.clear()
        self.is_trained = True

    def save_to_file(self, filename: str) -> None:
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
        mx.save(self._vectors, filename + "_xb")
        mx.save(self._ids, filename + "_ids")

    def clone(self) -> "FlatIndex":
        new_index = FlatIndex(self.d, self.metric_type)
        if self._vectors is not None and self._ids is not None:
            new_index._vectors = mx.copy(self._vectors)
            new_index._ids = mx.copy(self._ids)
            new_index.ntotal = int(self._vectors.shape[0])
            new_index._id_to_row = self._id_to_row.copy()
        return new_index
