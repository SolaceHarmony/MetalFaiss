import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform

class CenteringTransform(BaseLinearTransform):
    def __init__(self, d):
        super().__init__(d, d)
        self.mean = None
        
    def train(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        self.mean = mx.mean(vectors, axis=0)
        self.linear_transform = mx.eye(self.d_in)
        self._is_trained = True

    def apply(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        return vectors - self.mean

    def reverse_transform(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained") 
        vectors = mx.array(vectors, dtype=mx.float32)
        return vectors + self.mean
