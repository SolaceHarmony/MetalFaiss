"""
binary_flat_index.py - Binary flat index for MetalFaiss
"""

import numpy as np
import mlx.core as mx
from typing import List, Optional, Tuple, Union
from .binary_index import BaseBinaryIndex
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult, SearchRangeResult

class BinaryFlatIndex(BaseBinaryIndex):
    """Binary flat index.
    
    This index stores binary vectors in their original form and performs
    exhaustive search using Hamming distance.
    """
    
    def __init__(self, d: int):
        """Initialize binary flat index.
        
        Args:
            d: Dimension of binary vectors (must be multiple of 8)
        """
        super().__init__(d)
        self.codes = None
        
    def add(self, x: mx.array) -> None:
        """Add binary vectors to index.
        
        Args:
            x: Binary vectors to add (n, d)
        """
        if x.shape[1] != self.d:
            raise ValueError(f"Vector dimension {x.shape[1]} != index dimension {self.d}")
            
        # Convert to uint8
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)
            
        # Initialize or append codes
        if self.codes is None:
            self.codes = x
        else:
            self.codes = mx.concatenate([self.codes, x], axis=0)
            
        self.ntotal = len(self.codes)
        
    def search(
        self,
        x: mx.array,
        k: int,
        metric: Optional[MetricType] = None
    ) -> Tuple[mx.array, mx.array]:
        """Search for k nearest neighbors.
        
        Args:
            x: Query vectors (n, d)
            k: Number of nearest neighbors
            metric: Optional metric type (ignored for binary indices)
            
        Returns:
            distances: Hamming distances (n, k)
            indices: Indices of nearest neighbors (n, k)
        """
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} != index dimension {self.d}")
            
        if self.ntotal == 0:
            return (
                mx.zeros((len(x), k), dtype=np.int32),
                mx.zeros((len(x), k), dtype=np.int32)
            )
            
        # Convert to uint8
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)
            
        # Compute Hamming distances
        distances = mx.zeros((len(x), self.ntotal), dtype=np.int32)
        for i in range(self.d):
            distances += mx.not_equal(
                x[:, i:i+1],
                self.codes[:, i].reshape(1, -1)
            )
            
        # Get top k
        if k > self.ntotal:
            k = self.ntotal
        indices = mx.argsort(distances, axis=1)[:, :k]
        distances = mx.take_along_axis(distances, indices, axis=1)
        
        return distances, indices
        
    def range_search(
        self,
        x: mx.array,
        radius: float,
        metric: Optional[MetricType] = None
    ) -> SearchRangeResult:
        """Search for vectors within Hamming radius.
        
        Args:
            x: Query vectors (n, d)
            radius: Maximum Hamming distance
            metric: Optional metric type (ignored for binary indices)
            
        Returns:
            SearchRangeResult containing distances and indices
        """
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} != index dimension {self.d}")
            
        if self.ntotal == 0:
            return SearchRangeResult([], [], mx.array([0] * (len(x) + 1)))
            
        # Convert to uint8
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)
            
        # Compute Hamming distances
        distances = mx.zeros((len(x), self.ntotal), dtype=np.int32)
        for i in range(self.d):
            distances += mx.not_equal(
                x[:, i:i+1],
                self.codes[:, i].reshape(1, -1)
            )
            
        # Get matches within radius
        matches = []
        for i in range(len(x)):
            mask = distances[i] <= radius
            matches.append([
                (int(d), int(j))
                for d, j in zip(
                    distances[i][mask].numpy(),
                    mx.arange(self.ntotal)[mask].numpy()
                )
            ])
            
        # Build lims array
        lims = [0]
        for m in matches:
            lims.append(lims[-1] + len(m))
            
        # Flatten matches
        flat_distances = []
        flat_indices = []
        for m in matches:
            for d, j in m:
                flat_distances.append(mx.array([d]))
                flat_indices.append(mx.array([j]))
                
        return SearchRangeResult(
            flat_distances,
            flat_indices,
            mx.array(lims)
        )
        
    def reconstruct(self, idx: Union[int, mx.array]) -> mx.array:
        """Reconstruct vectors from their indices.
        
        Args:
            idx: Vector indices to reconstruct
            
        Returns:
            Reconstructed vectors
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= self.ntotal:
                raise ValueError(f"Index {idx} out of bounds [0, {self.ntotal})")
            return self.codes[idx:idx+1]
        else:
            if mx.min(idx) < 0 or mx.max(idx) >= self.ntotal:
                raise ValueError(f"Indices out of bounds [0, {self.ntotal})")
            return self.codes[idx]
            
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self.codes = None