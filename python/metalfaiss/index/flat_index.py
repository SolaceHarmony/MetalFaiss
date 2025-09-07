from typing import List, Optional, ClassVar
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
        self._metric_type = metric_type
        self._vectors: Optional[mx.array] = None
        self._is_trained = True  # legacy flag
        self.is_trained = True   # base flag
        
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
        """Train is a no-op for flat index."""
        pass  # Flat index doesn't need training
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs (ignored in flat index)
        """
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        if self._vectors is None:
            self._vectors = x
        else:
            self._vectors = mx.concatenate([self._vectors, x])
        self.ntotal = len(self._vectors)
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if self._vectors is None:
            raise RuntimeError("Index is empty")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        k = min(k, self.ntotal)
        
        # Compute distances matrix (lower is better), except for INNER_PRODUCT
        mt = self.metric_type
        if mt == MetricType.L2:
            distances = pairwise_L2sqr(x, self._vectors)
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.L1:
            distances = pairwise_L1(x, self._vectors)
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.Linf:
            distances = pairwise_Linf(x, self._vectors)
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.Canberra:
            distances = pairwise_extra_distances(x, self._vectors, "Canberra")
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.BrayCurtis:
            distances = pairwise_extra_distances(x, self._vectors, "BrayCurtis")
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.JensenShannon:
            distances = pairwise_extra_distances(x, self._vectors, "JensenShannon")
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.Jaccard:
            distances = pairwise_jaccard(x, self._vectors)
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        elif mt == MetricType.INNER_PRODUCT:
            sims = mx.matmul(x, self._vectors.T)
            distances = -sims
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)
        else:
            # Fallback to L2
            distances = pairwise_L2sqr(x, self._vectors)
            idx = mx.argsort(distances, axis=1)[:, :k]
            vals = mx.take_along_axis(distances, idx, axis=1)

        return vals, idx
        
    def xb(self) -> List[List[float]]:
        """Get stored vectors.
        
        Returns:
            List of stored vectors
        """
        if self._vectors is None:
            return []
        return self._vectors.tolist()
    @property
    def metric_type(self) -> MetricType:
        return self._metric_type
