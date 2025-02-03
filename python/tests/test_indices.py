import unittest
import mlx.core as mx
import numpy as np
from ..index.FlatIndex import FlatIndex
from ..index.IVFFlatIndex import IVFFlatIndex
from ..PQIndex import PQIndex
from ..PreTransformIndex import PreTransformIndex

class TestIndices(unittest.TestCase):
    def setUp(self):
        self.d = 64
        self.n = 1000
        self.xs = mx.random.normal(shape=(self.n, self.d)).astype(mx.float32)
        self.query = mx.random.normal(shape=(1, self.d)).astype(mx.float32)

    def test_flat_index(self):
        index = FlatIndex(self.d)
        index.add(self.xs)
        indices, distances = index.search(self.query, k=5)
        self.assertEqual(len(indices), 5)
        self.assertTrue(mx.all(distances[:-1] <= distances[1:]))

    def test_ivf_flat_index(self):
        index = IVFFlatIndex(self.d, nlist=10)
        index.train(self.xs)
        index.add(self.xs)
        indices, distances = index.search(self.query, k=5)
        self.assertLessEqual(len(indices), 5)
        if len(indices) > 1:
            self.assertTrue(mx.all(distances[:-1] <= distances[1:]))

    def test_pre_transform_index(self):
        matrix = mx.random.normal(shape=(self.d, 32)).astype(mx.float32)
        base_index = FlatIndex(32)
        index = PreTransformIndex(base_index, matrix)
        index.add(self.xs)
        indices, distances = index.search(self.query, k=5)
        self.assertEqual(len(indices), 5)
        self.assertTrue(mx.all(distances[:-1] <= distances[1:]))
