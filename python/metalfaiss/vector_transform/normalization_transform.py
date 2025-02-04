import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform

class NormalizationTransform(BaseLinearTransform):
    def __init__(self, d, norm=2.0):
        super().__init__(d, d)
        self.norm = norm
        self._is_trained = True
        
    def train(self, vectors=None):
        pass  # No training needed
        
    def apply(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        norms = mx.sum(mx.abs(vectors) ** self.norm, axis=1) ** (1.0 / self.norm)
        norms = mx.expand_dims(norms, axis=1)
        return vectors / norms
        
    def reverse_transform(self, vectors):
        return self.apply(vectors)  # Same operation
