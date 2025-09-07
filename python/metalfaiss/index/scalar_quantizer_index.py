"""
scalar_quantizer_index.py - Index with scalar quantization of vectors
"""

from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult
from ..utils.sorting import mlx_topk
from ..index.index_error import IndexError, TrainingError

class ScalarQuantizerIndex(BaseIndex):
    """Index that uses scalar quantization for vector compression."""
    
    def __init__(self, d: int, qtype: str = 'QT_8bit'):
        """Initialize scalar quantizer index.
        
        Args:
            d: Vector dimension
            qtype: Quantizer type ('QT_8bit', 'QT_4bit', etc.)
        """
        super().__init__(d)
        self._qtype = qtype
        self._trained_vectors = None
        self._codes = None
        
    def train(self, xs: List[List[float]]) -> None:
        """Train the scalar quantizer.
        
        Args:
            xs: Training vectors
            
        Raises:
            TrainingError: If training fails
        """
        if not xs:
            raise ValueError("Empty training data")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        try:
            # Store training vectors for computing quantization parameters
            self._trained_vectors = x
            self._is_trained = True
        except Exception as e:
            raise TrainingError(f"Failed to train scalar quantizer: {e}")
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add and quantize vectors.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # TODO: Implement actual scalar quantization
        # For now, just store the raw vectors
        if self._codes is None:
            self._codes = x
        else:
            self._codes = mx.concatenate([self._codes, x])
            
        self._ntotal += len(x)
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if self._codes is None:
            raise RuntimeError("Index is empty")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        k = min(k, self.ntotal)
        
        # TODO: Implement efficient distance computation with quantized vectors
        # For now, compute exact distances
        if self.metric_type == MetricType.L2:
            distances = mx.sum(mx.square(mx.subtract(x.reshape(len(x), 1, -1), self._codes)), axis=2)
        else:
            distances = mx.negative(mx.matmul(x, self._codes.T))
            
        values, indices = mlx_topk(distances, k, axis=1, largest=False)
            
        return SearchResult(distances=values, indices=indices)
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self._codes = None
        self._trained_vectors = None
