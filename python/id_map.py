import mlx.core as mx
import numpy as np

class IDMap:
    def __init__(self, subIndex):
        self.subIndex = subIndex
        self.id_map = []

    def add(self, vectors, ids=None):
        if ids is not None:
            self.id_map.extend(ids)
        else:
            self.id_map.extend(range(len(self.id_map), len(self.id_map) + len(vectors)))
        self.subIndex.add(vectors)

    def search(self, query, k):
        indices, distances = self.subIndex.search(query, k)
        return [self.id_map[i] for i in indices], distances

class IDMap2(IDMap):
    def __init__(self, subIndex):
        super().__init__(subIndex)
        self.vectors = {}

    def add(self, vectors, ids):
        for vec, id in zip(vectors, ids):
            self.vectors[id] = vec
        super().add(vectors, ids)

    def reconstruct(self, id):
        return self.vectors[id]
