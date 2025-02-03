import mlx
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
        xs = mlx.array(xs, dtype=np.float32)
        n = xs.shape[0]
        centroids = xs[mlx.random.choice(n, self.m, replace=False)]
        for _ in range(100):
            distances = mlx.linalg.norm(xs[:, np.newaxis] - centroids, axis=2)
            labels = mlx.argmin(distances, axis=1)
            new_centroids = mlx.array([xs[labels == i].mean(axis=0) for i in range(self.m)])
            if mlx.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids

    def add(self, vectors):
        vectors = mlx.array(vectors, dtype=np.float32)
        self.vectors.extend(vectors)

    def search(self, query, k):
        query = mlx.array(query, dtype=np.float32)
        vectors = mlx.array(self.vectors, dtype=np.float32)
        if self.metric_type == 'l2':
            distances = mlx.linalg.norm(vectors - query, axis=1)
        elif self.metric_type == 'inner_product':
            distances = -mlx.dot(vectors, query)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        indices = mlx.argsort(distances)[:k]
        return indices, distances[indices]
