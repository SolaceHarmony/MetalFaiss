# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

from abc import ABC, abstractmethod
from typing import List

class BaseVectorTransform(ABC):
    """
    Base class for all vector transformations.
    
    This class defines the interface for vector transformations used in MetalFaiss.
    All transformations must implement the training and application methods.
    """
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the transform is trained and ready to use."""
        pass

    @property
    @abstractmethod
    def d_in(self) -> int:
        """Input dimension of the transformation."""
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        """Output dimension of the transformation."""
        pass

    @abstractmethod
    def train(self, vectors: List[List[float]]) -> None:
        """
        Train the transformation on a set of vectors.
        
        Args:
            vectors: Training vectors to learn the transformation from
        """
        pass

    @abstractmethod
    def apply(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Apply the transformation to vectors.
        
        Args:
            vectors: Input vectors to transform
            
        Returns:
            Transformed vectors
        """
        pass

    @abstractmethod
    def reverse_transform(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Apply the inverse transformation to vectors.
        
        Args:
            vectors: Transformed vectors to reverse
            
        Returns:
            Original vectors (approximate)
        """
        pass
