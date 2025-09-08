"""
binary_flat_index.py - Binary flat index for MetalFaiss
"""

import mlx.core as mx
from typing import List, Optional, Tuple, Union
from .binary_index import BaseBinaryIndex
from ..types.metric_type import MetricType
from ..faissmlx.device_guard import require_gpu
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
        require_gpu("BinaryFlatIndex.add")
        if x.shape[1] != self.d:
            raise ValueError(f"Vector dimension {x.shape[1]} != index dimension {self.d}")
            
        # Convert to uint8 (unconditionally to avoid dtype checks in hot path)
        x = x.astype(mx.uint8)
            
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
        require_gpu("BinaryFlatIndex.search")
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} != index dimension {self.d}")
            
        if self.ntotal == 0:
            return (
                mx.zeros((len(x), k), dtype=mx.int32),
                mx.zeros((len(x), k), dtype=mx.int32)
            )
            
        # Convert to uint8 (unconditionally to avoid dtype checks in hot path)
        x = x.astype(mx.uint8)
            
        # Compute Hamming distances
        distances = mx.zeros((len(x), self.ntotal), dtype=mx.int32)
        for i in range(self.d):
            inc = mx.not_equal(
                x[:, i:i+1],
                self.codes[:, i].reshape(1, -1)
            ).astype(mx.int32)
            distances = mx.add(distances, inc)
            
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
        require_gpu("BinaryFlatIndex.range_search")
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} != index dimension {self.d}")
            
        if self.ntotal == 0:
            return SearchRangeResult([], [], mx.array([0] * (len(x) + 1)))
            
        # Convert to uint8 (unconditionally to avoid dtype checks in hot path)
        x = x.astype(mx.uint8)
            
        # Compute Hamming distances
        distances = mx.zeros((len(x), self.ntotal), dtype=mx.int32)
        for i in range(self.d):
            inc = mx.not_equal(
                x[:, i:i+1],
                self.codes[:, i].reshape(1, -1)
            ).astype(mx.int32)
            distances = mx.add(distances, inc)
            
        # Get matches within radius
        flat_distances = []
        flat_indices = []
        lims = [0]
        rad = mx.array(radius, dtype=mx.int32)
        for i in range(len(x)):
            mask = mx.less_equal(distances[i], rad)
            d_arr = distances[i][mask]
            j_arr = mx.arange(self.ntotal, dtype=mx.int32)[mask]
            # Append as 1-element arrays without converting to Python scalars
            for p in range(int(d_arr.shape[0])):
                flat_distances.append(d_arr[p:p+1])
                flat_indices.append(j_arr[p:p+1])
            lims.append(lims[-1] + int(d_arr.shape[0]))

        return SearchRangeResult(
            flat_distances,
            flat_indices,
            mx.array(lims, dtype=mx.int32)
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
            lo_bad = mx.less(mx.min(idx), mx.array(0, dtype=idx.dtype))
            hi_bad = mx.greater_equal(mx.max(idx), mx.array(self.ntotal, dtype=idx.dtype))
            any_bad = bool(mx.any(mx.logical_or(lo_bad, hi_bad)).item())  # boundary-ok
            if any_bad:
                raise ValueError(f"Indices out of bounds [0, {self.ntotal})")
            return self.codes[idx]
            
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self.codes = None
