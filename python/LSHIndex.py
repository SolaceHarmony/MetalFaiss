import numpy as np
import mlx
import metal

class LSHIndex:
    def __init__(self, d, nbits, rotate_data=False, train_thresholds=False):
        self.d = d
        self.nbits = nbits
        self.rotate_data = rotate_data
        self.train_thresholds = train_thresholds
        self.hash_functions = self._generate_hash_functions()
        self.hash_table = {}

    def _generate_hash_functions(self):
        np.random.seed(0)
        return [np.random.randn(self.d) for _ in range(self.nbits)]

    def _hash_vector(self, vector):
        return tuple((np.dot(vector, h) > 0).astype(int) for h in self.hash_functions)

    def add(self, vectors):
        for vector in vectors:
            hash_value = self._hash_vector(vector)
            if hash_value not in self.hash_table:
                self.hash_table[hash_value] = []
            self.hash_table[hash_value].append(vector)

    def search(self, query, k):
        hash_value = self._hash_vector(query)
        candidates = self.hash_table.get(hash_value, [])
        if self.rotate_data:
            query = self._rotate_vector(query)
            candidates = [self._rotate_vector(c) for c in candidates]
        distances = [np.linalg.norm(query - c) for c in candidates]
        indices = np.argsort(distances)[:k]
        return indices, [distances[i] for i in indices]

    def _rotate_vector(self, vector):
        rotation_matrix = np.random.randn(self.d, self.d)
        return np.dot(vector, rotation_matrix)
