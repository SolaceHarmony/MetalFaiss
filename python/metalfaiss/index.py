# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

# TODO: Review implementations from:
# - faiss/Index.h
# - faiss/Index.cpp 
# - faiss/IndexFlat.h
# - faiss/IndexFlat.cpp
# - faiss/IndexFlatCodes.h
# - faiss/IndexFlatCodes.cpp

import mlx.core as mx
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from .metric_type import MetricType
from .range_search import RangeSearchResult
from .utils.search_result import SearchResult

FAISS_VERSION = (1, 10, 0)

class Index(ABC):
    """Abstract base class for vector indexes."""
    
    def __init__(self, d: int, metric_type: MetricType = MetricType.L2):
        """Initialize index.
        
        Args:
            d: Vector dimension
            metric_type: Distance metric type
        """
        self.d = d
        self.ntotal = 0
        self.verbose = False
        self.is_trained = True
        self.metric_type = metric_type
        self.metric_arg = 0.0

    def train(self, xs: List[List[float]]) -> None:
        """Train index on vectors (no-op by default).
        
        Args:
            xs: Training vectors
        """
        pass

    @abstractmethod
    def add(self, xs: List[List[float]]) -> None:
        """Add vectors to index.
        
        Args:
            xs: Vectors to add
        """
        pass

    def add_with_ids(self, xs: List[List[float]], ids: List[int]) -> None:
        """Add vectors with custom IDs.
        
        Args:
            xs: Vectors to add
            ids: Vector IDs
            
        Note:
            Default implementation ignores IDs and calls regular add().
            Subclasses that support custom IDs should override this method.
        """
        # Default implementation - just add vectors without considering IDs
        # Subclasses that support custom IDs should override this
        import warnings
        warnings.warn(
            "add_with_ids called on index that doesn't support custom IDs. "
            "Using regular add() instead.",
            UserWarning
        )
        self.add(xs)

    @abstractmethod 
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for k nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of neighbors
            
        Returns:
            SearchResult with distances and labels
        """
        pass

    def range_search(self, xs: List[List[float]], radius: float) -> RangeSearchResult:
        """Search for vectors within radius.
        
        Args:
            xs: Query vectors
            radius: Search radius
            
        Returns:
            Range search results
        """
        # Default implementation using regular search
        # Subclasses can override for more efficient implementations
        from .range_search import range_search_with_index
        return range_search_with_index(self, xs, radius)

    def assign(self, xs: List[List[float]], k: int = 1) -> List[int]:
        """Assign vectors to nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of neighbors
            
        Returns:
            Neighbor labels
        """
        result = self.search(xs, k)
        return result.labels[0]

    @abstractmethod
    def reset(self) -> None:
        """Reset the index."""
        pass

    def reconstruct(self, idx: int) -> mx.array:
        """Reconstruct vector at index.
        
        Args:
            idx: Vector index
            
        Returns:
            Reconstructed vector
            
        Note:
            Default implementation raises NotImplementedError.
            Only indexes that store full vectors can implement this.
        """
        raise NotImplementedError(
            "reconstruct not implemented. "
            "This index type does not support vector reconstruction."
        )

    def reconstruct_n(self, i0: int, ni: int) -> mx.array:
        """Reconstruct range of vectors.
        
        Args:
            i0: Start index
            ni: Number of vectors
            
        Returns:
            Reconstructed vectors
        """
        return mx.stack([self.reconstruct(i) for i in range(i0, i0 + ni)])

    def search_and_reconstruct(
        self,
        xs: List[List[float]], 
        k: int
    ) -> Tuple[SearchResult, List[List[List[float]]]]:
        """Search and reconstruct results.
        
        Args:
            xs: Query vectors
            k: Number of neighbors
            
        Returns:
            Tuple of (search results, reconstructed vectors)
        """
        result = self.search(xs, k)
        recons = []
        for query_labels in result.labels:
            query_recons = []
            for label in query_labels:
                if label != -1:
                    query_recons.append(self.reconstruct(label))
                else:
                    query_recons.append([0.0] * self.d)
            recons.append(query_recons)
        return result, recons
