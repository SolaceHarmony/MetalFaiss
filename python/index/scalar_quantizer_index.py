import mlx.core as mx
import numpy as np
from .index.BaseIndex import BaseIndex
from ..Errors import TrainingError

class ScalarQuantizerIndex(BaseIndex):
    def __init__(self, d: int, qtype: str = 'QT_8bit'):
        self.d = d
        self.qtype = qtype
        self._is_trained = False
        self.vmin = None
        self.vmax = None
        self.vectors = []
        
    def train(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        if len(vectors) == 0:
            raise TrainingError("Cannot train on empty dataset")
            
        # Compute range for quantization
        self.vmin = mx.min(vectors, axis=0)
        self.vmax = mx.max(vectors, axis=0)
        self._is_trained = True

    def add(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        # Quantize and store vectors
        scale = (self.vmax - self.vmin) / 255
        quantized = mx.floor((vectors - self.vmin) / scale)
        self.vectors.extend(quantized)

    def search(self, query, k):
        query = mx.array(query, dtype=mx.float32)
        if len(self.vectors) == 0:
            return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.float32)
            
        # Dequantize and compute distances
        scale = (self.vmax - self.vmin) / 255
        reconstructed = self.vmin + scale * mx.array(self.vectors)
        distances = mx.linalg.norm(query[:, None, :] - reconstructed, axis=2)
        
        indices = mx.argsort(distances, axis=1)[:, :k]
        final_distances = mx.take_along_axis(distances, indices, axis=1)
        return indices, final_distances
