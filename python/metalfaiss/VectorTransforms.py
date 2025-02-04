import mlx.core as mx
import numpy as np
from .vector_transform import BaseVectorTransform

class CenteringTransform(BaseVectorTransform):
    def __init__(self, d):
        super().__init__(d, d)
        self.mean = None

    def train(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        self.mean = mx.mean(vectors, axis=0)
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

class ITQMatrixTransform(BaseVectorTransform):
    def __init__(self, d):
        super().__init__(d, d)
        self.rotation_matrix = None
        self.is_orthonormal = False

    def train(self, vectors):
        self.rotation_matrix = mx.random.normal(0, 1, (self.d_in, self.d_out))
        self._is_trained = True

    def apply(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained") 
        vectors = mx.array(vectors, dtype=mx.float32)
        return mx.matmul(vectors, self.rotation_matrix)

    def reverse_transform(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        return mx.matmul(vectors, mx.linalg.pinv(self.rotation_matrix))

class NormalizationTransform(BaseVectorTransform):
    def __init__(self, d, norm=1.0):
        super().__init__(d, d)
        self.norm = norm
        self._is_trained = True

    def train(self, vectors):
        pass # Nothing to train

    def apply(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        norms = mx.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms * self.norm

    def reverse_transform(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        norms = mx.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / self.norm * norms
