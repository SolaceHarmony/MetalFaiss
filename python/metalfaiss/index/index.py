"""
index.py - Base index interface matching SwiftFaiss structure
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import mlx.core as mx
from ..types.metric_type import MetricType
from ..utils.range_search import RangeSearchResult
from ..utils.search_result import SearchResult
from .index_pointer import IndexPointer

class BaseIndex(ABC):
    """Base class for all index types.
    
    This abstract class defines the interface that all index implementations must support.
    It provides basic functionality for vector storage, searching, and management.
    """
    
    @abstractmethod
    def index_pointer(self) -> IndexPointer:
        """The underlying index pointer. Required property matching Swift protocol."""
        pass

    def __init__(self, d: int):
        """Initialize index.
        
        Args:
            d: Dimension of vectors to be indexed
        """
        self._d = d
        self._is_trained = False
        self._ntotal = 0
        self._metric_type = MetricType.L2
        self._verbose = False
        
    @property
    @abstractmethod
    def d(self) -> int:
        """Dimension of vectors in the index."""
        pass
        
    @property
    def is_trained(self) -> bool:
        """Whether the index has been trained."""
        return self._is_trained
        
    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        return self._ntotal
        
    @property
    def metric_type(self) -> MetricType:
        """Distance metric used by the index."""
        return self._metric_type
        
    @property
    def verbose(self) -> bool:
        """Whether to output verbose logging."""
        return self._verbose
        
    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value
        
    def assign(self, xs: List[List[float]], k: int) -> List[List[int]]:
        """Assign vectors to nearest centroids.
        
        Args:
            xs: Query vectors
            k: Number of assignments per vector
            
        Returns:
            List of k nearest centroid indices per vector
            
        Raises:
            IndexError: If assignment fails
        """
        raise NotImplementedError("assign not implemented")
        
    @abstractmethod
    def train(self, xs: List[List[float]]) -> None:
        """Train the index.
        
        Args:
            xs: Training vectors
            
        Raises:
            IndexError: If training fails
            ValueError: If input dimensions don't match
        """
        if not xs:
            raise ValueError("Empty training data")
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
        
    @abstractmethod
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs
            
        Raises:
            IndexError: If addition fails
            RuntimeError: If index is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
        self._ntotal += len(x)
        
    @abstractmethod
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors to return
            
        Returns:
            SearchResult containing distances and labels
            
        Raises:
            IndexError: If search fails
        """
        pass
        
    def search_range(self, xs: List[List[float]], radius: float) -> RangeSearchResult:
        """Search for vectors within radius.
        
        Args:
            xs: Query vectors
            radius: Search radius
            
        Returns:
            RangeSearchResult containing matches within radius
            
        Raises:
            IndexError: If search fails
        """
        raise NotImplementedError("search_range not implemented")
        
    def reconstruct(self, key: int) -> mx.array:
        """Reconstruct vector from storage.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
            
        Raises:
            NotImplementedError: If reconstruction not supported
        """
        raise NotImplementedError("reconstruct not implemented")
        
    def reset(self) -> None:
        """Remove all vectors from the index."""
        self._ntotal = 0
        self._is_trained = False
        
    def save_to_file(self, filename: str) -> None:
        """Save index to file.
        
        Args:
            filename: Path to save the index
            
        Raises:
            IndexError: If save fails
        """
        raise NotImplementedError("save_to_file not implemented")
        
    def clone(self) -> 'BaseIndex':
        """Create a deep copy of the index.
        
        Returns:
            New index instance with same data
            
        Raises:
            IndexError: If clone fails
        """
        raise NotImplementedError("clone not implemented")
