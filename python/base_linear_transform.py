from abc import ABC, abstractmethod
import mlx.core as mx
from typing import List

class BaseLinearTransform(ABC):
    """Base class for linear transformations."""
    
    def __init__(self, d: int):
        """Initialize transform.
        
        Args:
            d: Input dimension
        """
        self._d = d
        self._is_trained = False
        
    @property
    def d(self) -> int:
        """Input dimension."""
        return self._d
        
    @property
    def is_trained(self) -> bool:
        """Whether transform has been trained."""
        return self._is_trained
        
    def train(self, xs: List[List[float]]) -> None:
        """Train the transformation.
        
        Args:
            xs: Training vectors
        """
        raise NotImplementedError
        
    def apply(self, xs: List[List[float]]) -> mx.array:
        """Apply transformation to vectors.
        
        Args:
            xs: Input vectors
            
        Returns:
            Transformed vectors
        """
        raise NotImplementedError
