import mlx.core as mx
import numpy as np
from .index.BaseIndex import BaseIndex
from .Errors import IndexError

class ProductQuantizerIndex(BaseIndex):
    def __init__(self, d: int, M: int, nbits: int = 8):
        if d % M != 0:
            raise IndexError(f"Dimension {d} not divisible by number of subquantizers {M}")
            
        self.d = d
        self.M = M  # Number of subquantizers
        self.nbits = nbits
        self.dsub = d // M  # Dimension of each subvector
        self.ksub = 1 << nbits  # Number of centroids for each subquantizer
        self._is_trained = False
        self.centroids = []  # List of centroids for each subquantizer
        self.codes = []  # Encoded vectors

    def train(self, vectors):
        vectors = mx.array(vectors, dtype=mx.float32)
        # Split vectors into M subvectors
        subvectors = vectors.reshape(-1, self.M, self.dsub)
        
        # Train each subquantizer using k-means
        self.centroids = []
        for m in range(self.M):
            # Select random initial centroids
            perm = mx.random.permutation(len(vectors))[:self.ksub]
            centroids = subvectors[perm, m]
            
            # k-means iterations
            for _ in range(10):
                distances = mx.linalg.norm(
                    subvectors[:, m, None, :] - centroids[None, :, :],
                    axis=2
                )
                labels = mx.argmin(distances, axis=1)
                new_centroids = []
                for k in range(self.ksub):
                    cluster = subvectors[labels == k, m]
                    if len(cluster) > 0:
                        new_centroids.append(mx.mean(cluster, axis=0))
                    else:
                        new_centroids.append(centroids[k])
                centroids = mx.stack(new_centroids)
                
            self.centroids.append(centroids)
            
        self._is_trained = True

    # ...existing code for add() and search() methods...
