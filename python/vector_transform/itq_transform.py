import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform
from .pca_matrix_transform import PCAMatrixTransform
from .itq_matrix_transform import ITQMatrixTransform

class ITQTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out, do_pca=True):
        super().__init__(d_in, d_out)
        self.do_pca = do_pca
        self.pca = None if not do_pca else PCAMatrixTransform(d_in, d_out)
        self.itq = ITQMatrixTransform(d_out)
        
    def train(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        
        if self.do_pca:
            self.pca.train(vectors)
            vectors = self.pca.apply(vectors)
            
        self.itq.train(vectors)
        self._is_trained = True
        
    def apply(self, vectors):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        vectors = mx.array(vectors, dtype=mx.float32)
        
        if self.do_pca:
            vectors = self.pca.apply(vectors)
            
        return self.itq.apply(vectors)
