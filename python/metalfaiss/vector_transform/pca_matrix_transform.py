import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform

class PCAMatrixTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out, eigen_power=0, random_rotation=False):
        super().__init__(d_in, d_out)
        self.eigen_power = eigen_power
        self.random_rotation = random_rotation
        self.mean = None
        
    def train(self, vectors):
        vectors = mx.eval(vectors)
        
        # Center the data
        self.mean = mx.eval(mx.mean(vectors, axis=0))
        centered = mx.eval(vectors - self.mean)

        # Compute covariance matrix and force evaluation
        cov = mx.eval(mx.matmul(centered.T, centered) / (len(vectors) - 1))
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = mx.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = mx.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Apply eigen power if specified
        if self.eigen_power != 0:
            eigenvalues = mx.power(eigenvalues, self.eigen_power)
            eigenvectors *= mx.sqrt(eigenvalues)

        # Select top d_out components
        self.linear_transform = eigenvectors[:, :self.d_out]
        
        if self.random_rotation:
            # Apply random rotation while preserving orthogonality
            R = mx.random.normal(0, 1, (self.d_out, self.d_out))
            Q, _ = mx.linalg.qr(R)
            self.linear_transform = mx.matmul(self.linear_transform, Q)
            
        self._is_trained = True

    def apply(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        return mx.matmul(vectors - self.mean, self.linear_transform)

    def reverse_transform(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        return mx.matmul(vectors, mx.linalg.pinv(self.linear_transform)) + self.mean
