import numpy as np
import mlx
import metal

class PreTransformIndex:
    def __init__(self, vector_transform, sub_index):
        self.vector_transform = vector_transform
        self.sub_index = sub_index

    def train(self, xs):
        transformed_xs = self.vector_transform.apply(xs)
        self.sub_index.train(transformed_xs)

    def add(self, vectors):
        transformed_vectors = self.vector_transform.apply(vectors)
        self.sub_index.add(transformed_vectors)

    def search(self, query, k):
        transformed_query = self.vector_transform.apply(query)
        return self.sub_index.search(transformed_query, k)
