import mlx
import metal

class FlatIndex:
    def __init__(self, d, metric_type='l2'):
        self.d = d
        self.metric_type = metric_type
        self.vectors = []

    def add(self, vectors):
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
