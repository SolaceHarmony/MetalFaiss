"""
search_result.py - Search result classes for MetalFaiss
"""

from dataclasses import dataclass
from typing import List, Optional
import mlx.core as mx
import numpy as np

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
        
    def to_numpy(self) -> tuple:
        """Convert to numpy arrays.
        
        Returns:
            Tuple of (distances, indices) as numpy arrays
        """
        return (
            self.distances.numpy(),
            self.indices.numpy()
        )
        
    @classmethod
    def from_numpy(cls, distances: np.ndarray, indices: np.ndarray) -> 'SearchResult':
        """Create from numpy arrays.
        
        Args:
            distances: Distances array
            indices: Indices array
            
        Returns:
            SearchResult object
        """
        return cls(
            distances=mx.array(distances),
            indices=mx.array(indices)
        )

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
        
    def to_numpy(self) -> tuple:
        """Convert to numpy arrays.
        
        Returns:
            Tuple of (distances list, indices list, lims array) as numpy
        """
        return (
            [d.numpy() for d in self.distances],
            [i.numpy() for i in self.indices],
            self.lims.numpy()
        )
        
    @classmethod
    def from_numpy(
        cls,
        distances: List[np.ndarray],
        indices: List[np.ndarray],
        lims: np.ndarray
    ) -> 'SearchRangeResult':
        """Create from numpy arrays.
        
        Args:
            distances: List of distances arrays
            indices: List of indices arrays
            lims: Limits array
            
        Returns:
            SearchRangeResult object
        """
        return cls(
            distances=[mx.array(d) for d in distances],
            indices=[mx.array(i) for i in indices],
            lims=mx.array(lims)
        )
        
    def get_total_size(self) -> int:
        """Get total number of results across all queries."""
        return int(self.lims[-1])
