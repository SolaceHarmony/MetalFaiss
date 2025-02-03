from typing import List, Optional, ClassVar
import mlx.core as mx
from .base_index import BaseIndex
from .search_result import SearchResult
from ..metric_type import MetricType
from ..errors import IndexError

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
        super().__init__(d)
        self._metric_type = metric_type
        self._vectors: Optional[mx.array] = None
        self._is_trained = True  # Flat index is always trained
        
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
        self._ntotal = len(self._vectors)
        
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
        distances = []
        labels = []
        
        for query in x:
            if self.metric_type == MetricType.L2:
                dists = mx.sum((self._vectors - query) ** 2, axis=1)
            else:  # Inner product
                dists = -mx.sum(self._vectors * query, axis=1)
                
            idx = mx.argsort(dists)[:k]
            distances.append(dists[idx].tolist())
            labels.append(idx.tolist())
            
        return SearchResult(distances=distances, labels=labels)
        
    def xb(self) -> List[List[float]]:
        """Get stored vectors.
        
        Returns:
            List of stored vectors
        """
        if self._vectors is None:
            return []
        return self._vectors.tolist()
