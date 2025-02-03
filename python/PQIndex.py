import numpy as np
import metal

class PQIndex:
    def __init__(self, d, m, nbits=8, metric_type='l2'):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.metric_type = metric_type
        self.centroids = None
        self.vectors = []

    def train(self, xs):
        xs = np.array(xs, dtype=np.float32)
        n = xs.shape[0]
        centroids = xs[np.random.choice(n, self.m, replace=False)]
        for _ in range(100):
            distances = np.linalg.norm(xs[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([xs[labels == i].mean(axis=0) for i in range(self.m)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids

    def add(self, vectors):
        vectors = np.array(vectors, dtype=np.float32)
        self.vectors.extend(vectors)

    def search(self, query, k):
        query = np.array(query, dtype=np.float32)
        vectors = np.array(self.vectors, dtype=np.float32)
        if self.metric_type == 'l2':
            distances = np.linalg.norm(vectors - query, axis=1)
        elif self.metric_type == 'inner_product':
            distances = -np.dot(vectors, query)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]
