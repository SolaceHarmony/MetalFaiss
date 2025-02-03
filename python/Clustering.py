import mlx
import metal

class KMeansClustering:
    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.centroids = None

    def train(self, xs):
        xs = mlx.array(xs, dtype=np.float32)
        n = xs.shape[0]
        centroids = xs[mlx.random.choice(n, self.k, replace=False)]
        for _ in range(100):
            distances = mlx.linalg.norm(xs[:, np.newaxis] - centroids, axis=2)
            labels = mlx.argmin(distances, axis=1)
            new_centroids = mlx.array([xs[labels == i].mean(axis=0) for i in range(self.k)])
            if mlx.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids

    def predict(self, xs):
        xs = mlx.array(xs, dtype=np.float32)
        distances = mlx.linalg.norm(xs[:, np.newaxis] - self.centroids, axis=2)
        return mlx.argmin(distances, axis=1)
