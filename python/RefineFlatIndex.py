import mlx.core as mx
import numpy as np
from .index.BaseIndex import BaseIndex
from .index.FlatIndex import FlatIndex

class RefineFlatIndex(BaseIndex):
    def __init__(self, base_index, k_factor=1.0):
        self.base_index = base_index
        self.k_factor = k_factor
        self._d = base_index.d
        self._is_trained = False
        
    @property
    def d(self):
        return self._d

    @property 
    def is_trained(self):
        return self._is_trained

    def train(self, x):
        self._is_trained = True

    def add(self, vectors):
        self.base_index.add(vectors)

    def search(self, query, k):
        # First get base results
        base_D, base_I = self.base_index.search(query, int(k * self.k_factor))
        
        # Refine distances by recomputing exact distances
        query = mx.array(query, dtype=mx.float32)
        if len(base_I) == 0:
            return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.float32)
            
        refined_D = mx.linalg.norm(
            query[:, None, :] - self.base_index.get_vectors(base_I)[None, :, :],
            axis=2
        )
        
        # Get top k after refinement
        indices = mx.argsort(refined_D, axis=1)[:, :k]
        final_D = mx.take_along_axis(refined_D, indices, axis=1)
        final_I = mx.take_along_axis(base_I, indices, axis=1) 
        
        return final_I, final_D
