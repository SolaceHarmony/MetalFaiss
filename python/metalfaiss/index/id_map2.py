from typing import List, Optional, Dict, Tuple
import mlx.core as mx
import numpy as np
from .base_index import BaseIndex
from .id_map import IDMap

class IDMap2(IDMap):
    """
    Index wrapper that stores a mapping between vector IDs and their vectors.
    Extends IDMap with vector storage functionality for reconstruction.
    """
    
    def __init__(self, sub_index: BaseIndex):
        """
        Initialize IDMap2 index.
        
        Args:
            sub_index: Base index to wrap
        """
        super().__init__(sub_index)
        self.vectors: Dict[int, mx.array] = {}
        self.d = sub_index.d
        self._is_trained = False
        
    def add(self, vectors: mx.array, ids: Optional[mx.array] = None) -> None:
        """
        Add vectors with optional IDs to the index.
        
        Args:
            vectors: Vectors to add
            ids: Optional vector IDs. If None, uses sequential IDs.
        """
        if ids is None:
            ids = mx.add(
                mx.arange(len(vectors)),
                mx.array(len(self.id_map), dtype=mx.int32)
            )
            
        # Store vectors for reconstruction
        for vec, id in zip(vectors, ids):
            self.vectors[int(id)] = vec
            
        super().add(vectors, ids)
        
    def reconstruct(self, key: int) -> mx.array:
        """
        Reconstruct vector from its ID.
        
        Args:
            key: Vector ID
            
        Returns:
            Reconstructed vector
            
        Raises:
            KeyError: If key not found
        """
        if key not in self.vectors:
            raise KeyError(f"No vector stored for ID {key}")
        return self.vectors[key]
        
    def reconstruct_batch(self, keys: List[int]) -> mx.array:
        """
        Reconstruct multiple vectors from their IDs.
        
        Args:
            keys: List of vector IDs
            
        Returns:
            Array of reconstructed vectors
            
        Raises:
            KeyError: If any key not found
        """
        vectors = []
        for key in keys:
            vectors.append(self.reconstruct(key))
        return mx.stack(vectors)
        
    def train(self, x: mx.array) -> None:
        """Train underlying index."""
        self.sub_index.train(x)
        self._is_trained = True
        
    @property 
    def is_trained(self) -> bool:
        """Whether index is trained."""
        return self._is_trained
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self.vectors.clear()
        self._is_trained = False
