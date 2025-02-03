import numpy as np
import metal

class IVFFlatIndex:
    def __init__(self, d, nlist, metric_type='l2'):
        self.d = d
        self.nlist = nlist
        self.metric_type = metric_type
        self.quantizer = FlatIndex(d, metric_type)
        self.centroids = None
        self.vectors = [[] for _ in range(nlist)]

    def train(self, xs):
        xs = np.array(xs, dtype=np.float32)
        n = xs.shape[0]
        centroids = xs[np.random.choice(n, self.nlist, replace=False)]
        for _ in range(100):
            distances = np.linalg.norm(xs[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([xs[labels == i].mean(axis=0) for i in range(self.nlist)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids

    def add(self, vectors):
        vectors = np.array(vectors, dtype=np.float32)
        distances = np.linalg.norm(vectors[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        for i, label in enumerate(labels):
            self.vectors[label].append(vectors[i])

    def search(self, query, k):
        query = np.array(query, dtype=np.float32)
        distances = np.linalg.norm(self.centroids - query, axis=1)
        closest_centroid = np.argmin(distances)
        vectors = np.array(self.vectors[closest_centroid], dtype=np.float32)
        if self.metric_type == 'l2':
            distances = np.linalg.norm(vectors - query, axis=1)
        elif self.metric_type == 'inner_product':
            distances = -np.dot(vectors, query)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]
