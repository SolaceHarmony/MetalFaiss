from abc import ABC, abstractmethod
import mlx.core as mx
import numpy as np

class BaseVectorTransform(ABC):
    def __init__(self, d_in: int, d_out: int):
        self._d_in = d_in
        self._d_out = d_out
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_out(self) -> int:
        return self._d_out

    @abstractmethod
    def train(self, vectors: mx.array) -> None:
        """Train the transform on input vectors"""
        vectors = mx.eval(vectors) # Force evaluation
        pass

    @abstractmethod
    def apply(self, vectors: mx.array) -> mx.array:
        """Apply the transform to input vectors"""
        vectors = mx.eval(vectors)  # Force evaluation
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
        pass

    @abstractmethod
    def reverse_transform(self, vectors: mx.array) -> mx.array:
        """Reverse the transform"""
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before reverse transform")
        pass
