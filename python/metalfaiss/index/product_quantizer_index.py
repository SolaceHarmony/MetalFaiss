"""
product_quantizer_index.py - Product Quantizer Index implementation using MLX

This implements a vector index based on Product Quantization, providing efficient
storage and search through vector compression.
"""

import mlx.core as mx
from typing import List, Optional, Tuple
from .base_index import BaseIndex
from .product_quantizer import ProductQuantizer
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult

class ProductQuantizerIndex(BaseIndex):
    """Index based on Product Quantizer.
    
    Vectors are approximated using PQ codes for efficient storage and search.
    """
    
    def __init__(self, d: int, M: int, nbits: int = 8, metric_type: MetricType = MetricType.L2):
        """Initialize PQ index.
        
        Args:
            d: Dimension of input vectors
            M: Number of subquantizers
            nbits: Number of bits per subquantizer index
            metric_type: Distance metric to use
        """
        super().__init__(d)
        self._metric_type = metric_type
        self.pq = ProductQuantizer(d, M, nbits)
        self._codes = None
        
    def train(self, xs: List[List[float]]) -> None:
        """Train the product quantizer.
        
        Args:
            xs: Training vectors
        """
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        self.pq.train(x)
        self._is_trained = True
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add and encode vectors.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Compute PQ codes
        codes = self.pq.compute_codes(x)
        
        # Store codes
        if self._codes is None:
            self._codes = codes
        else:
            self._codes = mx.concatenate([self._codes, codes])
            
        self._ntotal += len(x)
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        k = min(k, self.ntotal)
        n = len(x)
        
        # Compute distances between queries and database vectors
        if self.metric_type == MetricType.L2:
            # For L2, compute per-subquantizer distances then sum
            x_reshaped = x.reshape(n, self.pq.M, self.pq.dsub)
            distances = mx.zeros((n, self.ntotal))
            
            for m in range(self.pq.M):
                # Compute distances to centroids for this subquantizer
                diff = x_reshaped[:, m, None, :] - self.pq.centroids[m]  # Shape: (n, ksub, dsub)
                subdist = mx.sum(diff * diff, axis=2)  # Shape: (n, ksub)
                
                # Look up distances for codes
                distances += subdist[mx.arange(n)[:, None], self._codes[:, m]]
                
        else:  # Inner product
            # For IP, compute per-subquantizer inner products then sum
            x_reshaped = x.reshape(n, self.pq.M, self.pq.dsub)
            distances = mx.zeros((n, self.ntotal))
            
            for m in range(self.pq.M):
                # Compute inner products with centroids
                ip = mx.matmul(x_reshaped[:, m], self.pq.centroids[m].T)  # Shape: (n, ksub)
                
                # Look up inner products for codes
                distances += ip[mx.arange(n)[:, None], self._codes[:, m]]
            distances = -distances  # Convert to distances
            
        # Get k nearest neighbors
        values, indices = mx.topk(-distances, k, axis=1)
        values = -values
        
        return SearchResult(
            distances=values.tolist(),
            labels=indices.tolist()
        )
        
    def reconstruct(self, key: int) -> List[float]:
        """Reconstruct vector from its PQ code.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
        """
        if key < 0 or key >= self.ntotal:
            raise ValueError(f"Invalid key {key}")
            
        return self.pq.decode(self._codes[key:key+1])[0].tolist()
