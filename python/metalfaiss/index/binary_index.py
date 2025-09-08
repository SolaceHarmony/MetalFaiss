"""
binary_index.py - Base class for binary indices

This implements the base functionality for binary indices, which store and
search binary vectors using Hamming distance. Binary vectors are stored as
uint8 arrays where each component is 0 or 1.

Original: faiss/IndexBinary.h
"""

import mlx.core as mx
from typing import List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult
from ..vector_transform.binary_transform import BaseBinaryTransform
from ..errors import InvalidArgumentError

def hamming_distance_table() -> mx.array:
    """Create lookup table for Hamming weight of bytes."""
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = bin(i).count('1')
    return mx.array(table, dtype=mx.uint8)

# Global Hamming weight lookup table
HAMMING_TABLE = hamming_distance_table()

class BaseBinaryIndex(ABC):
    """Base class for binary indices.
    
    Binary indices store vectors of 0/1 bits and search using Hamming
    distance. They can optionally use binary transforms to preprocess
    vectors.
    """
    
    def __init__(self, d: int):
        """Initialize binary index.
        
        Args:
            d: Dimension in bits
        """
        if d <= 0:
            raise InvalidArgumentError("Dimension must be positive")
            
        self.d = d  # Dimension in bits
        self._ntotal = 0  # Number of indexed vectors
        self._is_trained = False
        
        # Optional transform applied to vectors before indexing
        self.transform: Optional[BaseBinaryTransform] = None
        
    @property
    def is_trained(self) -> bool:
        """Whether index is trained."""
        return self._is_trained
        
    @property
    def ntotal(self) -> int:
        """Number of indexed vectors."""
        return self._ntotal
        
    @property
    def metric_type(self) -> MetricType:
        """Distance metric (always Hamming for binary indices)."""
        return MetricType.HAMMING
        
    def train(self, xs: List[List[int]]) -> None:
        """Train index on representative vectors.
        
        Default implementation validates vectors and trains transform
        if present.
        
        Args:
            xs: Training vectors (each component must be 0 or 1)
        """
        self._validate_binary(xs)
        
        if self.transform is not None:
            self.transform.train(xs)
            xs = self.transform.apply(xs)
            
        self._train(xs)
        self._is_trained = True
        
    @abstractmethod
    def _train(self, xs: List[List[int]]) -> None:
        """Train index implementation.
        
        Args:
            xs: Training vectors (already transformed if applicable)
        """
        pass
        
    def add(self, xs: List[List[int]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to index.
        
        Args:
            xs: Vectors to add (each component must be 0 or 1)
            ids: Optional vector IDs
            
        Raises:
            RuntimeError: If index not trained
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        self._validate_binary(xs)
        
        if self.transform is not None:
            xs = self.transform.apply(xs)
            
        self._add(xs, ids)
        
    @abstractmethod
    def _add(
        self,
        xs: List[List[int]],
        ids: Optional[List[int]] = None
    ) -> None:
        """Add vectors to index implementation.
        
        Args:
            xs: Vectors to add (already transformed if applicable)
            ids: Optional vector IDs
        """
        pass
        
    def search(self, xs: List[List[int]], k: int) -> SearchResult:
        """Search for nearest neighbors by Hamming distance.
        
        Args:
            xs: Query vectors (each component must be 0 or 1)
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
            
        Raises:
            RuntimeError: If index not trained or empty
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        if self.ntotal == 0:
            raise RuntimeError("Index is empty")
            
        self._validate_binary(xs)
        
        if self.transform is not None:
            xs = self.transform.apply(xs)
            
        return self._search(xs, k)
        
    @abstractmethod
    def _search(self, xs: List[List[int]], k: int) -> SearchResult:
        """Search implementation.
        
        Args:
            xs: Query vectors (already transformed if applicable)
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        pass
        
    def reconstruct(self, key: int) -> mx.array:
        """Reconstruct vector from storage.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
            
        Raises:
            RuntimeError: If index empty
            ValueError: If key invalid
        """
        if self.ntotal == 0:
            raise RuntimeError("Index is empty")
            
        if key < 0 or key >= self.ntotal:
            raise ValueError(f"Invalid key {key}")
            
        return self._reconstruct(key)
        
    @abstractmethod
    def _reconstruct(self, key: int) -> mx.array:
        """Reconstruct implementation.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
        """
        pass
        
    def reset(self) -> None:
        """Reset index to empty state."""
        self._ntotal = 0
        
    def _validate_binary(self, xs: List[List[int]]) -> None:
        """Validate that vectors are binary (0/1 values).
        
        Args:
            xs: Vectors to validate
            
        Raises:
            ValueError: If vectors are not binary or wrong dimension
        """
        if not xs:
            raise ValueError("Empty vector list")
            
        if len(xs[0]) != self.d:
            raise ValueError(
                f"Expected {self.d} input dimensions, got {len(xs[0])}"
            )
            
        for x in xs:
            if not all(v in (0, 1) for v in x):
                raise ValueError("Vectors must contain only 0/1 values")
                
    @staticmethod
    def hamming_distances(x: mx.array, y: mx.array) -> mx.array:
        """Compute Hamming distances between vectors.
        
        Uses lookup table for efficient Hamming weight computation.
        
        Args:
            x: First vectors (n, d) of uint8
            y: Second vectors (m, d) of uint8
            
        Returns:
            Distances (n, m) of uint32
        """
        # XOR vectors
        xor = x[:, None, :] ^ y[None, :, :]  # Shape: (n, m, d)
        
        # Look up Hamming weights
        return mx.sum(
            HAMMING_TABLE[xor],
            axis=2,
            dtype=mx.uint32
        )  # Shape: (n, m)
