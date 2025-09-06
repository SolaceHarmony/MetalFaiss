"""
itq.py - Iterative Quantization transform for MetalFaiss (MLX-only)
"""

import mlx.core as mx
from typing import Optional
from .base_vector_transform import BaseVectorTransform
from .pca_matrix import PCAMatrixTransform

class ITQTransform(BaseVectorTransform):
    """Iterative Quantization transform.
    
    This transform applies PCA followed by iterative quantization to learn
    a rotation matrix that minimizes quantization error when converting
    to binary codes.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        n_iter: int = 50,
        random_rotation: bool = True,
        seed: Optional[int] = None,
        key: Optional[object] = None
    ):
        """Initialize ITQ transform.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension (default: same as input)
            n_iter: Number of iterations
            random_rotation: Whether to apply random rotation before PCA
            seed: Random seed (default: None)
        """
        # Initialize base with resolved output dimension
        resolved_d_out = d_out if d_out is not None else d_in
        super().__init__(d_in, resolved_d_out)
        self.n_iter = n_iter
        self.random_rotation = random_rotation
        self.seed = seed
        self._key = key if key is not None else (mx.random.key(int(seed)) if seed is not None else None)
        
        self.pca = None
        self.rotation_matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform.
        
        Args:
            x: Training vectors (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Apply PCA first
        self.pca = PCAMatrixTransform(
            self.d_in,
            self.d_out,
            random_rotation=self.random_rotation
        )
        self.pca.train(x)
        v = self.pca.apply(x)
        
        # Initialize random rotation via SVD on random normal (orthogonal)
        if self._key is not None:
            kR, self._key = mx.random.split(self._key, num=2)
            R0 = mx.random.normal(shape=(self.d_out, self.d_out), key=kR).astype(mx.float32)
        else:
            R0 = mx.random.normal(shape=(self.d_out, self.d_out)).astype(mx.float32)
        U, _, Vt = mx.linalg.svd(R0, stream=mx.cpu)
        R = mx.matmul(U, Vt)
        
        # Iterative quantization
        for _ in range(self.n_iter):
            # Fix R, update B
            z = mx.matmul(v, R)
            B = mx.sign(z)
            
            # Fix B, update R
            C = mx.matmul(v.T, B)
            U2, _, Vt2 = mx.linalg.svd(C, stream=mx.cpu)
            R = mx.matmul(U2, Vt2)
            
        self.rotation_matrix = R
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply transform to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_out)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Apply PCA then rotation
        v = self.pca.apply(x)
        return mx.matmul(v, self.rotation_matrix)
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse transform to vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform output dimension {self.d_out}")
            
        # Apply inverse rotation then inverse PCA
        v = mx.matmul(x, mx.transpose(self.rotation_matrix))
        return self.pca.reverse_transform(v)
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained
