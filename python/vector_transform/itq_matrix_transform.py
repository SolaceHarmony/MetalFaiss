import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform

class ITQMatrixTransform(BaseLinearTransform):
    def __init__(self, d):
        super().__init__(d, d)
        self.is_orthonormal = False
        self.niter = 50
        
    def train(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        
        # Initialize with PCA
        cov = mx.matmul(vectors.T, vectors) / (len(vectors) - 1)
        eigenvalues, eigenvectors = mx.linalg.eigh(cov)
        
        # Sort by eigenvalues descending
        idx = mx.argsort(-eigenvalues)
        self.linear_transform = eigenvectors[:, idx]
        
        # Iterative quantization step
        for _ in range(self.niter):
            # Project data
            projected = mx.matmul(vectors, self.linear_transform)
            
            # Compute binary codes
            Z = mx.sign(projected)
            
            # Update rotation
            U, _, Vt = mx.linalg.svd(mx.matmul(Z.T, projected))
            self.linear_transform = mx.matmul(self.linear_transform, mx.matmul(U, Vt))
            
        self._is_trained = True
