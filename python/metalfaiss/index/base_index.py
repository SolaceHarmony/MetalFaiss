"""
base_index.py - Base class for all indices
"""

from ..utils.search_result import SearchResult, SearchRangeResult
from ..types.metric_type import MetricType
from typing import Optional, List, Tuple, Union
import mlx.core as mx

class BaseIndex:
    """Base class for all indices."""
    
    def __init__(self, d: int, metric: MetricType = MetricType.L2):
        """Initialize base index.
        
        Args:
            d: Dimension of vectors
            metric: Distance metric to use
        """
        self.d = d
        self.metric = metric
        self.is_trained = False
        self.ntotal = 0
        
    def train(self, x: mx.array) -> None:
        """Train the index.
        
        Args:
            x: Training vectors (n, d)
        """
        if x.shape[1] != self.d:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != index dimension {self.d}")
        self.is_trained = True
        
    def add(self, x: mx.array) -> None:
        """Add vectors to the index.
        
        Args:
            x: Vectors to add (n, d)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        if x.shape[1] != self.d:
            raise ValueError(f"Vector dimension {x.shape[1]} != index dimension {self.d}")
        self.ntotal += x.shape[0]
        
    def search(self, x: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        """Search for nearest neighbors.
        
        Args:
            x: Query vectors (n, d)
            k: Number of nearest neighbors
            
        Returns:
            distances: Distances to nearest neighbors (n, k)
            indices: Indices of nearest neighbors (n, k)
        """
        raise NotImplementedError
        
    def range_search(self, x: mx.array, radius: float) -> SearchRangeResult:
        """Search for vectors within radius.
        
        Args:
            x: Query vectors (n, d)
            radius: Search radius
            
        Returns:
            SearchRangeResult containing distances and indices
        """
        raise NotImplementedError
        
    def reset(self) -> None:
        """Reset the index."""
        self.ntotal = 0
    
    # GPU-only project: keep a no-op `.to_gpu` for compatibility
    def to_gpu(self, resources=None):  # type: ignore[override]
        return self
        
    def __len__(self) -> int:
        """Get number of vectors in index."""
        return self.ntotal
