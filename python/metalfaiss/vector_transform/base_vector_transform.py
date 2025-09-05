"""
base_vector_transform.py - Base classes for vector transforms

This implements the base vector transform functionality from FAISS,
providing a common interface for all transforms.

Original: faiss/VectorTransform.h
"""

from abc import ABC, abstractmethod
import mlx.core as mx
import numpy as np
from typing import List, Optional
from ..errors import InvalidArgumentError

class BaseVectorTransform(ABC):
    """Base class for all vector transforms.
    
    This matches FAISS's VectorTransform class, providing a common interface
    for transformations applied to sets of vectors.
    """
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize transform.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        self._d_in = d_in
        self._d_out = d_out
        self._is_trained = False
        
    @property
    def is_trained(self) -> bool:
        """Whether transform is trained."""
        return self._is_trained
        
    @property
    def d_in(self) -> int:
        """Input dimension."""
        return self._d_in
        
    @property
    def d_out(self) -> int:
        """Output dimension."""
        return self._d_out
        
    def train(self, xs: List[List[float]]) -> None:
        """Train transform on representative vectors.
        
        Default implementation does nothing since most transforms
        don't require training.
        
        Args:
            xs: Training vectors
        """
        pass
        
    def apply(self, xs: List[List[float]]) -> List[List[float]]:
        """Apply transform to vectors.
        
        Args:
            xs: Input vectors
            
        Returns:
            Transformed vectors
            
        Raises:
            RuntimeError: If transform not trained
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        x = mx.array(xs)
        xt = mx.zeros((len(x), self.d_out))
        self.apply_noalloc(x, xt)
        return xt.tolist()
        
    @abstractmethod
    def apply_noalloc(self, x: mx.array, xt: mx.array) -> None:
        """Apply transform with pre-allocated output.
        
        Args:
            x: Input vectors (n, d_in)
            xt: Output buffer (n, d_out)
            
        Raises:
            RuntimeError: If transform not trained
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
    def reverse_transform(
        self,
        xs: List[List[float]]
    ) -> List[List[float]]:
        """Reverse transform (may be approximate).
        
        Args:
            xs: Input vectors
            
        Returns:
            Reverse transformed vectors
            
        Raises:
            RuntimeError: If transform not trained
            NotImplementedError: If reverse transform not supported
        """
        raise NotImplementedError(
            "Reverse transform not implemented for this transform"
        )
        
    @abstractmethod
    def check_identical(self, other: 'BaseVectorTransform') -> None:
        """Check if transforms are identical.
        
        Args:
            other: Transform to compare with
            
        Raises:
            ValueError: If transforms are not identical
        """
        if (
            self.d_in != other.d_in or
            self.d_out != other.d_out or
            self.is_trained != other.is_trained
        ):
            raise ValueError("Transforms have different parameters")

class BaseLinearTransform(BaseVectorTransform):
    """Base class for linear transforms.
    
    This implements y = Ax + b where:
    - A is a d_out x d_in matrix
    - b is an optional d_out bias vector
    - x is the input vector
    - y is the output vector
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        have_bias: bool = False
    ):
        """Initialize linear transform.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
            have_bias: Whether to use bias term
        """
        super().__init__(d_in, d_out)
        self.have_bias = have_bias
        self.is_orthonormal = False
        
        # Initialize transform parameters
        self.A: Optional[mx.array] = None  # Transform matrix
        self.b: Optional[mx.array] = None  # Bias vector
        
    def apply_noalloc(self, x: mx.array, xt: mx.array) -> None:
        """Apply linear transform y = Ax + b.
        
        Args:
            x: Input vectors (n, d_in)
            xt: Output buffer (n, d_out)
            
        Raises:
            RuntimeError: If transform not trained
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if self.A is None:
            raise RuntimeError("Transform matrix not initialized")
            
        # Apply matrix multiply
        mx.scatter(xt, mx.arange(len(xt)), mx.matmul(x, self.A.T))
        
        # Add bias if present
        if self.have_bias and self.b is not None:
            xt += self.b
            
    def transform_transpose(
        self,
        y: mx.array,
        x: mx.array
    ) -> None:
        """Apply transpose transform x = A^T(y - b).
        
        Args:
            y: Input vectors (n, d_out)
            x: Output buffer (n, d_in)
            
        Raises:
            RuntimeError: If transform not trained
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if self.A is None:
            raise RuntimeError("Transform matrix not initialized")
            
        # Subtract bias if present
        if self.have_bias and self.b is not None:
            y = y - self.b
            
        # Apply transposed matrix multiply
        mx.scatter(x, mx.arange(len(x)), mx.matmul(y, self.A))
        
    def reverse_transform(
        self,
        xs: List[List[float]]
    ) -> List[List[float]]:
        """Reverse transform (only if orthonormal).
        
        Args:
            xs: Input vectors
            
        Returns:
            Reverse transformed vectors
            
        Raises:
            RuntimeError: If transform not trained
            ValueError: If transform not orthonormal
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if not self.is_orthonormal:
            raise ValueError(
                "Reverse transform only supported for orthonormal matrices"
            )
            
        y = mx.array(xs)
        x = mx.zeros((len(y), self.d_in))
        self.transform_transpose(y, x)
        return x.tolist()
        
    def set_is_orthonormal(self) -> None:
        """Check if matrix A is orthonormal."""
        if self.A is None:
            raise RuntimeError("Transform matrix not initialized")
            
        # Compute A^T * A
        ATA = mx.matmul(self.A.T, self.A)
        
        # Should be identity matrix
        I = mx.eye(self.d_in)
        self.is_orthonormal = mx.all(mx.abs(ATA - I) < 1e-5)
        
    def check_identical(self, other: BaseVectorTransform) -> None:
        """Check if transforms are identical.
        
        Args:
            other: Transform to compare with
            
        Raises:
            ValueError: If transforms are not identical
        """
        super().check_identical(other)
        
        if not isinstance(other, BaseLinearTransform):
            raise ValueError("Not a linear transform")
            
        if (
            self.have_bias != other.have_bias or
            self.is_orthonormal != other.is_orthonormal or
            (self.A is None) != (other.A is None) or
            (self.A is not None and not mx.all(self.A == other.A)) or
            (self.have_bias and self.b is not None and
             not mx.all(self.b == other.b))
        ):
            raise ValueError("Linear transforms are not identical")
