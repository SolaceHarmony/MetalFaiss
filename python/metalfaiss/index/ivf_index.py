"""
ivf_index.py - Base class for IVF (Inverted File) indexes
"""

from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from .flat_index import FlatIndex
from ..types.metric_type import MetricType
from ..index.index_error import IndexError

class IVFIndex(BaseIndex):
    """Base class for all IVF (Inverted File) indexes.
    
    IVF indexes use a coarse quantizer to partition the vector space and
    store vectors in inverted lists for efficient search.
    """
    
    def __init__(self, quantizer: FlatIndex, d: int, nlist: int):
        """Initialize IVF index.
        
        Args:
            quantizer: Coarse quantizer (typically a FlatIndex)
            d: Vector dimension
            nlist: Number of inverted lists (partitions)
        """
        super().__init__(d)
        self._quantizer = quantizer
        self._nlist = nlist
        self._nprobe = 1  # Number of lists to probe during search
        self._invlists = [[] for _ in range(nlist)]
        
    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist
        
    @property
    def nprobe(self) -> int:
        """Number of lists to probe during search."""
        return self._nprobe
        
    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value < 1:
            raise ValueError("nprobe must be positive")
        self._nprobe = value
        
    @property
    def quantizer(self) -> FlatIndex:
        """The coarse quantizer used by this index."""
        return self._quantizer
        
    def train(self, xs: List[List[float]]) -> None:
        """Train the index.
        
        This trains both the coarse quantizer and any sub-quantizers.
        
        Args:
            xs: Training vectors
        """
        if not xs:
            raise ValueError("Empty training data")
            
        # Train quantizer first
        self._quantizer.train(xs)
        self._is_trained = True
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Assign vectors to lists using quantizer
        assignments = self._quantizer.search(xs, 1)
        for i, label in enumerate(assignments.labels):
            self._invlists[label[0]].append((ids[i] if ids else i, x[i]))
            
        self._ntotal += len(x)
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self._invlists = [[] for _ in range(self._nlist)]
        self._quantizer.reset()
