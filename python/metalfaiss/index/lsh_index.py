import mlx.core as mx
from .base_index import BaseIndex
from ..utils.search_result import SearchResult

class LSHIndex(BaseIndex):
    def __init__(self, d: int, nbits: int = 64):
        self.d = d
        self.nbits = nbits
        self._is_trained = False
        self.rotation_matrix = None
        self.vectors = []
        self.bits = []

    def train(self, vectors):
        # Generate random rotation matrix for LSH
        vectors = mx.array(vectors, dtype=mx.float32)
        self.rotation_matrix = mx.random.normal(shape=(self.d, self.nbits))
        self._is_trained = True

    def add(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        # Compute hash bits
        projections = mx.matmul(vectors, self.rotation_matrix)
        bits = projections > 0
        self.bits.extend(bits)
        self.vectors.extend(vectors)

    def search(self, query, k):
        query = mx.array(query, dtype=mx.float32)
        if len(self.bits) == 0:
            return SearchResult(distances=mx.zeros((len(query), 0), dtype=mx.float32),
                                labels=mx.zeros((len(query), 0), dtype=mx.int32))

        # Compute query hash and hamming distances
        query_proj = mx.matmul(query, self.rotation_matrix) > 0
        hamming_dists = mx.sum(query_proj[:, None, :] != self.bits, axis=2)
        
        # Get top k nearest
        indices = mx.argsort(hamming_dists, axis=1)[:, :k]
        distances = mx.take_along_axis(hamming_dists, indices, axis=1)
        return SearchResult(distances=distances, labels=indices)
