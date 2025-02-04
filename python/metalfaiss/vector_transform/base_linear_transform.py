from abc import ABC, abstractmethod
import mlx.core as mx
from typing import List
from ..base_vector_transform import BaseVectorTransform

class BaseLinearTransform(BaseVectorTransform):
    """Base class for linear transformations."""
    
    def __init__(self, d_in: int, d_out: int):
        super().__init__(d_in, d_out)
        self.linear_transform = None
        self.is_orthonormal = False
        self.have_bias = False
        
    def make_orthonormal(self):
        """Make the transform orthonormal"""
        if self.linear_transform is not None:
            with mx.stream():
                Q, _ = mx.linalg.qr(self.linear_transform)
                self.linear_transform = Q
                mx.eval(self.linear_transform)
            self.is_orthonormal = True
            
    def transform_transpose(self, vectors: List[List[float]]) -> mx.array:
        """Apply transpose of transform to vectors"""
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        with mx.stream():
            result = mx.matmul(vectors, self.linear_transform.T)
            mx.eval(result)
        return result
        
    def apply(self, vectors: List[List[float]]) -> mx.array:
        """Apply transform to vectors"""
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        with mx.stream():
            result = mx.matmul(vectors, self.linear_transform)
            mx.eval(result)
        return result
        
    def reverse_transform(self, vectors: List[List[float]]) -> mx.array:
        """Apply inverse transform"""
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        with mx.stream():
            if self.is_orthonormal:
                # For orthonormal matrices, transpose is inverse
                result = mx.matmul(vectors, self.linear_transform.T)
            else:
                # Otherwise use pseudo-inverse
                result = mx.matmul(vectors, mx.linalg.pinv(self.linear_transform))
            mx.eval(result)
        return result
