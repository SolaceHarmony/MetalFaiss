"""
binary_hnsw_index.py - Binary HNSW (Hierarchical Navigable Small World) index

This implements HNSW for binary vectors, using Hamming distance for neighbor
selection and optimizing storage and computation for binary data.

Original: faiss/IndexBinaryHNSW.h
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple, Callable
from .binary_index import BaseBinaryIndex
from .hnsw import HNSW, HNSWStats
from ..utils.search_result import SearchResult
from ..errors import InvalidArgumentError
from ..faissmlx.device_guard import require_gpu

class BinaryHNSWIndex(BaseBinaryIndex):
    """Binary HNSW index.
    
    This adapts the HNSW graph structure for binary vectors, using Hamming
    distance for neighbor selection and optimizing storage and computation
    for binary data.
    """
    
    def __init__(
        self,
        d: int,
        M: int = 32,
        efConstruction: int = 40,
        efSearch: int = 16
    ):
        """Initialize binary HNSW index.
        
        Args:
            d: Dimension in bits
            M: Number of neighbors per layer (except layer 0 which has 2*M)
            efConstruction: Size of dynamic candidate list during construction
            efSearch: Size of dynamic candidate list during search
        """
        super().__init__(d)
        self.hnsw = HNSW(M)
        self.hnsw.efConstruction = efConstruction
        self.hnsw.efSearch = efSearch
        
        # Storage for binary vectors
        self.xb: Optional[mx.array] = None  # Shape: (n, d) of uint8
        self.ids: Optional[mx.array] = None  # Shape: (n,) of int32
        
        # Distance computer function
        self._dist_computer = self._make_dist_computer()
        
    def _make_dist_computer(self) -> Callable[[int, int], float]:
        """Create distance computer function for HNSW.
        
        Returns:
            Function that computes Hamming distance between stored vectors
        """
        def dist_computer(i: int, j: int) -> float:
            """Compute Hamming distance between vectors i and j."""
            if self.xb is None:
                raise RuntimeError("No vectors added to index")
                
            # Use hamming_distances from base class
            dist = self.hamming_distances(
                self.xb[i:i+1],
                self.xb[j:j+1]
            )[0, 0]
            return float(dist)
            
        return dist_computer
        
    def _train(self, xs: List[List[int]]) -> None:
        """Binary HNSW requires no training."""
        pass
        
    def _add(
        self,
        xs: List[List[int]],
        ids: Optional[List[int]] = None
    ) -> None:
        require_gpu("BinaryHNSWIndex.add")
        """Add binary vectors to the index.
        
        Args:
            xs: Vectors to add (each component must be 0 or 1)
            ids: Optional vector IDs
        """
        x = mx.array(xs, dtype=mx.uint8)
        n = len(x)
        
        # Initialize storage if needed
        if self.xb is None:
            self.xb = x
        else:
            self.xb = mx.concatenate([self.xb, x])
            
        # Store or generate IDs
        if ids is not None:
            id_array = mx.array(ids)
            if self.ids is None:
                self.ids = id_array
            else:
                self.ids = mx.concatenate([self.ids, id_array])
        else:
            if self.ids is None:
                self.ids = mx.arange(self._ntotal, self._ntotal + n)
            else:
                self.ids = mx.concatenate([
                    self.ids,
                    mx.arange(self._ntotal, self._ntotal + n)
                ])
                
        # Add vectors to HNSW graph
        for i in range(n):
            # Generate random level
            level = self.hnsw.random_level()
            
            # Add vertex with connections
            self.hnsw.add_vertex(
                self._ntotal + i,
                level,
                self._dist_computer
            )
            
        self._ntotal += n
        
    def _search(self, xs: List[List[int]], k: int) -> SearchResult:
        require_gpu("BinaryHNSWIndex.search")
        """Search for nearest neighbors by Hamming distance.
        
        Uses HNSW graph to efficiently find approximate nearest neighbors
        in Hamming space.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if self.xb is None:
            raise RuntimeError("No vectors added to index")
            
        x = mx.array(xs, dtype=mx.uint8)
        n = len(x)
        
        # Search each query
        out_vals = mx.zeros((n, k), dtype=mx.float32)
        out_ids = mx.zeros((n, k), dtype=mx.int32)
        inf = mx.divide(mx.ones((), dtype=mx.float32), mx.zeros((), dtype=mx.float32))
        
        for i in range(n):
            # Create distance computer for this query
            def query_dist_computer(j: int) -> float:
                dist = self.hamming_distances(
                    x[i:i+1],
                    self.xb[j:j+1]
                )[0, 0]
                return float(dist.astype(mx.float32).item())  # boundary-ok
                
            # Search HNSW graph
            results = self.hnsw.search(
                i,
                max(k, self.hnsw.efSearch),
                query_dist_computer
            )
            
            # Get top k results
            if len(results) > k:
                results = results[:k]
                
            # Split distances and indices
            if results:
                dists, idx = zip(*results)
            else:
                dists, idx = [], []
                
            # Convert to MLX arrays, pad with MLX scalars
            L = len(dists)
            if L > 0:
                dv = mx.array(dists, dtype=mx.float32)
                iv = mx.array(idx, dtype=mx.int32)
            else:
                dv = mx.zeros((0,), dtype=mx.float32)
                iv = mx.zeros((0,), dtype=mx.int32)
            if L < k:
                pad = k - L
                dv = mx.concatenate([dv, mx.broadcast_to(inf, (pad,))])
                iv = mx.concatenate([iv, mx.full((pad,), -1, dtype=mx.int32)])
            # Map to external IDs if present
            if self.ids is not None:
                mask = mx.greater_equal(iv, mx.zeros_like(iv))
                mapped = mx.where(mask, mx.take(self.ids, mx.maximum(iv, mx.zeros_like(iv))), mx.full_like(iv, -1))
                iv = mapped.astype(mx.int32)
            out_vals = mx.scatter(out_vals, mx.array([i]), dv.reshape((1, k)))
            out_ids = mx.scatter(out_ids, mx.array([i]), iv.reshape((1, k)))

        return SearchResult(distances=out_vals, indices=out_ids)
        
    def _reconstruct(self, key: int) -> mx.array:
        """Reconstruct vector from storage.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
            
        Raises:
            RuntimeError: If no vectors added
            ValueError: If key invalid
        """
        if self.xb is None or self.ids is None:
            raise RuntimeError("No vectors added to index")
            
        # Find positions for ID using MLX only
        eq = mx.equal(self.ids, mx.array(key, dtype=self.ids.dtype))
        idxs = mx.where(eq)[0]
        if int(idxs.shape[0]) == 0:
            raise ValueError(f"Invalid key {key}")
        return self.xb[idxs[0]]
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self.xb = None
        self.ids = None
        self.hnsw = HNSW(self.hnsw.M)
        
    @property
    def M(self) -> int:
        """Number of neighbors per layer."""
        return self.hnsw.M
        
    @property
    def efConstruction(self) -> int:
        """Size of dynamic candidate list during construction."""
        return self.hnsw.efConstruction
        
    @efConstruction.setter
    def efConstruction(self, value: int) -> None:
        if value <= 0:
            raise ValueError("efConstruction must be positive")
        self.hnsw.efConstruction = value
        
    @property
    def efSearch(self) -> int:
        """Size of dynamic candidate list during search."""
        return self.hnsw.efSearch
        
    @efSearch.setter
    def efSearch(self, value: int) -> None:
        if value <= 0:
            raise ValueError("efSearch must be positive")
        self.hnsw.efSearch = value
