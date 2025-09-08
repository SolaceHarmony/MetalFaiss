"""
hnsw_index.py - HNSW index implementation using MLX

This implements the HNSW index following the FAISS implementation, combining
the HNSW graph structure with MLX-optimized vector storage and computations.
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple
from .base_index import BaseIndex
from .hnsw import HNSW, HNSWStats
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult
from ..distances import pairwise_L2sqr
from ..faissmlx.device_guard import require_gpu

class HNSWIndex(BaseIndex):
    """HNSW index with optimized vector storage."""
    
    def __init__(
        self,
        d: int,
        M: int = 32,
        metric_type: MetricType = MetricType.L2
    ):
        """Initialize HNSW index.
        
        Args:
            d: Dimension of vectors
            M: Number of neighbors per layer (except layer 0 which has 2*M)
            metric_type: Distance metric to use
        """
        super().__init__(d)
        self._metric_type = metric_type
        
        # HNSW graph structure
        self.hnsw = HNSW(M)
        # Precompute MLX +inf scalar for sentinels
        self._inf = mx.divide(mx.ones((), dtype=mx.float32), mx.zeros((), dtype=mx.float32))
        
        # Vector storage using MLX array
        self._vectors: Optional[mx.array] = None
        
        # Parameters matching FAISS defaults
        self.hnsw.efConstruction = 40
        self.hnsw.efSearch = 16
        self.hnsw.check_relative_distance = True
        self.hnsw.search_bounded_queue = True
        
    def _compute_distances_batch(
        self,
        query: mx.array,
        vectors: mx.array,
        batch_size: int = 4
    ) -> mx.array:
        """Compute distances for a batch of vectors efficiently.
        
        Args:
            query: Query vector (d,)
            vectors: Database vectors (n, d)
            batch_size: Size of batches for computation
            
        Returns:
            Distances array (n,)
        """
        n = len(vectors)
        distances = mx.zeros(n)
        
        # Process in batches
        for i in range(0, n, batch_size):
            batch = vectors[i:min(i + batch_size, n)]
            if self._metric_type == MetricType.L2:
                # Use broadcasting for L2 computation
                diff = mx.subtract(batch, query.reshape(1, -1))
                batch_dists = mx.sum(mx.multiply(diff, diff), axis=1)
            else:  # Inner product
                batch_dists = mx.negative(mx.matmul(batch, query.reshape(-1, 1)).reshape(-1))
            distances = mx.scatter(
                distances,
                mx.arange(i, min(i + batch_size, n)),
                batch_dists
            )
            
        return distances
        
    def _make_dist_computer(self, query: mx.array):
        """Create optimized distance computer function.
        
        This implements the FAISS distance computation strategy with:
        - Batch processing
        - Distance caching
        - MLX acceleration
        """
        if self._vectors is None:
            raise RuntimeError("Index is empty")
            
        # Cache for computed distances
        cache = {}
        
        def dist_computer(i: int, j: int) -> float:
            if i == -1 or j == -1:
                return float(self._inf.astype(mx.float32).item())  # boundary-ok
            if i >= len(self._vectors) or j >= len(self._vectors):
                return float(self._inf.astype(mx.float32).item())  # boundary-ok
                
            # Check cache
            cache_key = (min(i, j), max(i, j))
            if cache_key in cache:
                return cache[cache_key]
                
            # Compute distance
            vec1 = query if i == len(self._vectors) else self._vectors[i]
            vec2 = self._vectors[j]
            
            if self._metric_type == MetricType.L2:
                diff = mx.subtract(vec1, vec2)
                dist = float(mx.sum(mx.multiply(diff, diff)).astype(mx.float32).item())  # boundary-ok
            else:  # Inner product
                dist = float(mx.negative(mx.dot(vec1, vec2)).astype(mx.float32).item())  # boundary-ok
            
            # Cache result
            cache[cache_key] = dist
            return dist
            
        return dist_computer
        
    def train(self, xs: List[List[float]]) -> None:
        """Train is a no-op for HNSW index."""
        self.is_trained = True
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        require_gpu("HNSWIndex.add")
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs (ignored in HNSW)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Store vectors
        if self._vectors is None:
            self._vectors = x
        else:
            self._vectors = mx.concatenate([self._vectors, x])
            
        # Rebuild a simple level-0 kNN graph for current vectors
        self.ntotal += len(x)
        n = int(self._vectors.shape[0])
        M0 = self.hnsw.M0
        # Compute distances
        diff = self._vectors[:, None, :] - self._vectors[None, :, :]
        d2 = mx.sum(mx.square(diff), axis=2)
        # Mask self-distances with +inf
        inf = mx.divide(mx.ones((), dtype=mx.float32), mx.zeros((), dtype=mx.float32))
        eye = mx.eye(n, dtype=mx.float32)
        mask = mx.greater(eye, mx.zeros_like(eye))
        d2 = mx.where(mask, inf, d2)
        # Get top M0 neighbors for each row
        order = mx.argsort(d2, axis=1)[:, :M0]
        # Fill HNSW arrays
        self.hnsw.levels = mx.zeros((n,), dtype=mx.int32)
        # Offsets point into a flat neighbor array
        self.hnsw.offsets = mx.arange(0, n * M0, M0, dtype=mx.int64)
        self.hnsw.neighbors = order.reshape((n * M0,)).astype(mx.int32)
        self.hnsw.entry_point = 0
        self.hnsw.max_level = 0
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        require_gpu("HNSWIndex.search")
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

        # Array-native level-0 search per query using MLX; returns MLX arrays
        ef = max(self.hnsw.efSearch, k)

        @mx.custom_function
        def _one(q):
            return self.hnsw.search_level0_array(q, ef=ef, vectors=self._vectors, metric=self._metric_type, k=k)

        @_one.vmap
        def _one_vmap(inputs, axes):
            (Q,), (ax,) = inputs, axes
            # We only support batching on axis 0
            if ax is None:
                d, i = _one(Q)
                return (d, i), None
            n = int(Q.shape[0])
            Ds = []
            Is = []
            for iidx in range(n):
                d, ii = self.hnsw.search_level0_array(Q[iidx], ef=ef, vectors=self._vectors, metric=self._metric_type, k=k)
                Ds.append(d); Is.append(ii)
            return (mx.stack(Ds, axis=0), mx.stack(Is, axis=0)), 0

        D, I = mx.vmap(_one, in_axes=0, out_axes=(0, 0))(x)
        return SearchResult(distances=D, indices=I)
        
    def reconstruct(self, key: int) -> mx.array:
        """Reconstruct vector from storage.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
        """
        if self._vectors is None:
            raise RuntimeError("Index is empty")
            
        if key < 0 or key >= self.ntotal:
            raise ValueError(f"Invalid key {key}")
            
        return self._vectors[key]
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self._vectors = None
        self.hnsw = HNSW(self.hnsw.M)

class HNSWFlatIndex(HNSWIndex):
    """HNSW index with flat storage."""
    
    def __init__(self, d: int, M: int = 32, metric_type: MetricType = MetricType.L2):
        super().__init__(d, M, metric_type)

class HNSWPQIndex(HNSWIndex):
    """HNSW index with Product Quantizer storage."""
    
    def __init__(
        self,
        d: int,
        M: int = 32,
        pq_m: int = 8,
        pq_nbits: int = 8,
        metric_type: MetricType = MetricType.L2
    ):
        super().__init__(d, M, metric_type)
        # TODO: Implement PQ storage
        raise NotImplementedError("PQ storage not yet implemented")
