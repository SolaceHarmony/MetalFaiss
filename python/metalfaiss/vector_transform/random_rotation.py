"""
random_rotation.py - Random rotation transform for MetalFaiss (MLX-only)
"""

import mlx.core as mx
from typing import Optional
from .base_vector_transform import BaseVectorTransform
from ..faissmlx.qr import pure_mlx_qr

class RandomRotationTransform(BaseVectorTransform):
    """Random rotation transform.
    
    This transform applies a random orthogonal matrix to the input vectors.
    The matrix is generated using QR decomposition of a random matrix.
    """
    
    def __init__(self, d_in: int, d_out: Optional[int] = None, seed: Optional[int] = None, key: Optional[object] = None):
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
        self._key = key if key is not None else (mx.random.key(int(seed)) if seed is not None else None)
        self.rotation_matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by generating a random rotation matrix.
        
        Args:
            x: Training vectors (not used)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Generate random matrix using MLX PRNG
        if self._key is not None:
            kA, self._key = mx.random.split(self._key, num=2)
            A = mx.random.normal(shape=(self.d_in, self.d_out), key=kA).astype(mx.float32)
        else:
            A = mx.random.normal(shape=(self.d_in, self.d_out)).astype(mx.float32)
        # QR decomposition: prefer GPU-only kernel if available
        # MLX-only QR (Modified Gram–Schmidt)
        Q, R = pure_mlx_qr(A)
        Q = Q[:, : self.d_out]
        self.rotation_matrix = Q
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
