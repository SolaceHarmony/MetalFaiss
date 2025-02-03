import mlx
import metal

class RefineFlatIndex:
    def __init__(self, base_index, k_factor=1.0):
        self.base_index = base_index
        self.k_factor = k_factor
        self.refine_index = FlatIndex(base_index.d, base_index.metric_type)

    def train(self, xs):
        self.base_index.train(xs)
        self.refine_index.train(xs)

    def add(self, vectors):
        self.base_index.add(vectors)
        self.refine_index.add(vectors)

    def search(self, query, k):
        base_indices, base_distances = self.base_index.search(query, int(k * self.k_factor))
        refine_indices, refine_distances = self.refine_index.search(query, k)
        return refine_indices, refine_distances
