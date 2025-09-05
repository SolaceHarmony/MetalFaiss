"""
pca_matrix.py - PCA matrix transform for MetalFaiss
"""

import numpy as np
import mlx.core as mx
from typing import Optional, Tuple
from .base_vector_transform import BaseVectorTransform
from .random_rotation import RandomRotationTransform

class PCAMatrixTransform(BaseVectorTransform):
    """PCA matrix transform.
    
    This transform applies PCA dimensionality reduction to the input vectors.
    It can optionally apply random rotation before PCA.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        eigen_power: float = 0.0,
        random_rotation: bool = True,
        epsilon: float = 1e-5
    ):
        """Initialize PCA matrix transform.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension (default: same as input)
            eigen_power: Power to apply to eigenvalues (0 = no scaling)
            random_rotation: Whether to apply random rotation before PCA
            epsilon: Small value to add to eigenvalues for numerical stability
        """
        # Initialize base with resolved output dimension
        resolved_d_out = d_out if d_out is not None else d_in
        super().__init__(d_in, resolved_d_out)
        self.eigen_power = eigen_power
        self.random_rotation = random_rotation
        self.epsilon = epsilon
        
        self.mean = None
        self.pca_matrix = None
        self.random_rotation_matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by computing PCA.
        
        Args:
            x: Training vectors (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Center data
        self.mean = mx.mean(x, axis=0)
        x_centered = x - self.mean
        
        # Optional random rotation
        if self.random_rotation:
            rotator = RandomRotationTransform(self.d_in)
            rotator.train(x_centered)
            x_centered = rotator.apply(x_centered)
            self.random_rotation_matrix = rotator.rotation_matrix
            
        # Compute covariance matrix
        cov = mx.matmul(x_centered.T, x_centered) / (len(x) - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov.numpy())
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Apply eigenvalue scaling
        if self.eigen_power != 0:
            scale = np.power(eigenvalues[:self.d_out] + self.epsilon, 
                           self.eigen_power / 2)
            eigenvectors[:, :self.d_out] *= scale
            
        # Take needed components
        self.pca_matrix = mx.array(eigenvectors[:, :self.d_out])
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply PCA transform to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_out)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Center and optionally rotate
        x = x - self.mean
        if self.random_rotation:
            x = mx.matmul(x, self.random_rotation_matrix)
            
        # Apply PCA
        return mx.matmul(x, self.pca_matrix)
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse PCA transform to vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform output dimension {self.d_out}")
            
        # Inverse PCA
        x = mx.matmul(x, mx.transpose(self.pca_matrix))
        
        # Inverse rotation and uncenter
        if self.random_rotation:
            x = mx.matmul(x, mx.transpose(self.random_rotation_matrix))
        return x + self.mean
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained
