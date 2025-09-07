"""
search_result.py - Search result classes for MetalFaiss
"""

from dataclasses import dataclass
from typing import List, Optional
import mlx.core as mx

@dataclass
class SearchResult:
    """Result of k-nearest neighbor search.
    
    Attributes:
        distances: Distances to nearest neighbors (n, k)
        indices: Indices of nearest neighbors (n, k)
    """
    distances: mx.array
    indices: mx.array
    
    def __post_init__(self):
        """Validate search result."""
        if self.distances.shape != self.indices.shape:
            raise ValueError("Distances and indices must have same shape")
            
    def __len__(self) -> int:
        """Get number of queries."""
        return len(self.distances)
        
    def __getitem__(self, idx) -> 'SearchResult':
        """Get result for specific query."""
        return SearchResult(
            distances=self.distances[idx],
            indices=self.indices[idx]
        )
    
    # Compatibility alias for code paths that historically used 'labels'.
    # We keep this as a read-only view to avoid legacy container types.
    @property
    def labels(self) -> mx.array:
        return self.indices
        
    # NumPy conversion helpers intentionally omitted in pure MLX build.

@dataclass
class SearchRangeResult:
    """Result of range search.
    
    Attributes:
        distances: List of distances for each query
        indices: List of indices for each query
        lims: Limits array indicating start/end of each query's results
    """
    distances: List[mx.array]
    indices: List[mx.array]
    lims: mx.array
    
    def __post_init__(self):
        """Validate range search result."""
        if len(self.distances) != len(self.indices):
            raise ValueError("Must have same number of distance and index arrays")
        if len(self.lims) != len(self.distances) + 1:
            raise ValueError("Lims array must have length n_queries + 1")
            
    def __len__(self) -> int:
        """Get number of queries."""
        return len(self.distances)
        
    def __getitem__(self, idx) -> tuple:
        """Get result for specific query.
        
        Returns:
            Tuple of (distances, indices) for query
        """
        return (self.distances[idx], self.indices[idx])
        
    # NumPy conversion helpers intentionally omitted in pure MLX build.
        
    def get_total_size(self) -> int:
        """Get total number of results across all queries."""
        return int(self.lims[-1])
