import mlx
import metal

class IVFScalarQuantizerIndex:
    def __init__(self, d, nlist, quantizer_type='QT8bit', metric_type='l2', encode_residual=False):
        self.d = d
        self.nlist = nlist
        self.quantizer_type = quantizer_type
        self.metric_type = metric_type
        self.encode_residual = encode_residual
        self.quantizer = FlatIndex(d, metric_type)
        self.centroids = None
        self.vectors = [[] for _ in range(nlist)]

    def train(self, xs):
        xs = mlx.array(xs, dtype=mlx.core.eval.float32)
        n = xs.shape[0]
        centroids = xs[mlx.random.choice(n, self.nlist, replace=False)]
        for _ in range(100):
            distances = mlx.linalg.norm(xs[:, mlx.core.eval.newaxis] - centroids, axis=2)
            labels = mlx.argmin(distances, axis=1)
            new_centroids = mlx.array([mlx.core.eval(xs[labels == i].mean(axis=0)) for i in range(self.nlist)])
            if mlx.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids

    def add(self, vectors):
        vectors = mlx.array(vectors, dtype=mlx.core.eval.float32)
        distances = mlx.linalg.norm(vectors[:, mlx.core.eval.newaxis] - self.centroids, axis=2)
        labels = mlx.argmin(distances, axis=1)
        for i, label in enumerate(labels):
            self.vectors[label].append(vectors[i])

    def search(self, query, k):
        query = mlx.array(query, dtype=mlx.core.eval.float32)
        distances = mlx.linalg.norm(self.centroids - query, axis=1)
        closest_centroid = mlx.argmin(distances)
        vectors = mlx.array(self.vectors[closest_centroid], dtype=mlx.core.eval.float32)
        if self.metric_type == 'l2':
            distances = mlx.linalg.norm(vectors - query, axis=1)
        elif self.metric_type == 'inner_product':
            distances = -mlx.dot(vectors, query)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        indices = mlx.argsort(distances)[:k]
        return indices, distances[indices]
