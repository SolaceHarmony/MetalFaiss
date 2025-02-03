import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform

class RandomRotationMatrixTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        self.is_orthonormal = True
        
    def train(self, vectors=None):
        # Generate random rotation matrix
        R = mx.random.normal(0, 1, (self.d_in, self.d_out))
        Q, _ = mx.linalg.qr(R)  # Orthogonalize
        self.linear_transform = Q
        self._is_trained = True
