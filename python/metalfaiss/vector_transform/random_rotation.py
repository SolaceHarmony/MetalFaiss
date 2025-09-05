"""
random_rotation.py - Random rotation transform for MetalFaiss
"""

import numpy as np
import mlx.core as mx
from typing import Optional
from .base_vector_transform import BaseVectorTransform

class RandomRotationTransform(BaseVectorTransform):
    """Random rotation transform.
    
    This transform applies a random orthogonal matrix to the input vectors.
    The matrix is generated using QR decomposition of a random matrix.
    """
    
    def __init__(self, d_in: int, d_out: Optional[int] = None, seed: Optional[int] = None):
        """Initialize random rotation transform.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension (default: same as input)
            seed: Random seed (default: None)
        """
        # Initialize base with resolved output dimension
        resolved_d_out = d_out if d_out is not None else d_in
        super().__init__(d_in, resolved_d_out)
        self.seed = seed
        self.rotation_matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by generating a random rotation matrix.
        
        Args:
            x: Training vectors (not used)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Generate random matrix
        A = np.random.randn(self.d_in, self.d_out).astype('float32')
        
        # QR decomposition to get orthogonal matrix
        Q, R = np.linalg.qr(A)
        
        # Make sure Q is orthogonal
        Q = Q.astype('float32')
        Q = Q[:, :self.d_out]  # Take only needed columns
        
        # Store rotation matrix
        self.rotation_matrix = mx.array(Q)
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply random rotation to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Rotated vectors (n, d_out)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Apply rotation
        return mx.matmul(x, self.rotation_matrix)
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse random rotation to vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse rotated vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform output dimension {self.d_out}")
            
        # Apply inverse rotation (transpose since matrix is orthogonal)
        return mx.matmul(x, mx.transpose(self.rotation_matrix))
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained
