import mlx
import metal

class ScalarQuantizerIndex:
    def __init__(self, d, quantizer_type='QT8bit', metric_type='l2'):
        self.d = d
        self.quantizer_type = quantizer_type
        self.metric_type = metric_type
        self.quantizer = FlatIndex(d, metric_type)
        self.vectors = []

    def train(self, xs):
        xs = mlx.array(xs, dtype=mlx.core.eval.float32)
        self.quantizer.train(xs)

    def add(self, vectors):
        vectors = mlx.array(vectors, dtype=mlx.core.eval.float32)
        self.vectors.extend(vectors)
        self.quantizer.add(vectors)

    def search(self, query, k):
        query = mlx.array(query, dtype=mlx.core.eval.float32)
        distances = mlx.linalg.norm(self.vectors - query, axis=1)
        indices = mlx.argsort(distances)[:k]
        return indices, distances[indices]
