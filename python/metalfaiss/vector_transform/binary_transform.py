"""
binary_transform.py - Binary vector transforms for MetalFaiss
"""

import numpy as np
import mlx.core as mx
from typing import Optional, Tuple
from .base_vector_transform import BaseVectorTransform

class BaseBinaryTransform(BaseVectorTransform):
    """Base class for binary vector transforms."""
    
    def __init__(self, d_in: int, d_out: Optional[int] = None):
        """Initialize binary transform.
        
        Args:
            d_in: Input dimension (must be multiple of 8)
            d_out: Output dimension (default: same as input)
        """
        resolved_d_out = d_out if d_out is not None else d_in
        super().__init__(d_in, resolved_d_out)
        if d_in % 8 != 0:
            raise ValueError(f"Input dimension {d_in} must be multiple of 8")
        if self.d_out % 8 != 0:
            raise ValueError(f"Output dimension {self.d_out} must be multiple of 8")
            
    def apply(self, x: mx.array) -> mx.array:
        """Apply transform to binary vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_out)
        """
        raise NotImplementedError
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse transform to binary vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        raise NotImplementedError

class BinaryRotationTransform(BaseBinaryTransform):
    """Binary rotation transform.
    
    This transform applies a random binary rotation matrix to the input vectors.
    The rotation matrix is a random permutation matrix that preserves Hamming
    distances between vectors.
    """
    
    def __init__(self, d_in: int, seed: Optional[int] = None):
        """Initialize binary rotation transform.
        
        Args:
            d_in: Input dimension (must be multiple of 8)
            seed: Random seed (default: None)
        """
        super().__init__(d_in)
        self.seed = seed
        self.permutation = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by generating random permutation.
        
        Args:
            x: Training vectors (not used)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Generate random permutation
        self.permutation = mx.array(np.random.permutation(self.d_in))
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply binary rotation to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Rotated vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Apply permutation
        return x[:, self.permutation]
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse binary rotation to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Inverse rotated vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Apply inverse permutation
        inv_perm = mx.zeros_like(self.permutation)
        inv_perm[self.permutation] = mx.arange(self.d_in)
        return x[:, inv_perm]
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained

class BinaryMatrixTransform(BaseBinaryTransform):
    """Binary matrix transform.
    
    This transform applies a binary matrix multiplication to the input vectors.
    The matrix is trained to minimize reconstruction error while preserving
    Hamming distances between vectors.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        n_iter: int = 10,
        seed: Optional[int] = None
    ):
        """Initialize binary matrix transform.
        
        Args:
            d_in: Input dimension (must be multiple of 8)
            d_out: Output dimension (default: same as input)
            n_iter: Number of training iterations
            seed: Random seed (default: None)
        """
        super().__init__(d_in, d_out)
        self.n_iter = n_iter
        self.seed = seed
        self.matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by learning binary matrix.
        
        Args:
            x: Training vectors (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Initialize random matrix
        self.matrix = mx.array(
            np.random.randint(0, 2, (self.d_in, self.d_out)),
            dtype=np.uint8
        )
        
        # Iterate to minimize reconstruction error
        for _ in range(self.n_iter):
            # Forward pass
            y = self.apply(x)
            
            # Backward pass - update matrix
            grad = mx.matmul(x.T, y)
            self.matrix = (grad > grad.mean()).astype(np.uint8)
            
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply binary matrix transform to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_out)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Binary matrix multiplication
        y = mx.matmul(x, self.matrix)
        return (y > y.mean()).astype(np.uint8)
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse binary matrix transform to vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform output dimension {self.d_out}")
            
        # Binary matrix multiplication with transpose
        y = mx.matmul(x, self.matrix.T)
        return (y > y.mean()).astype(np.uint8)
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained
