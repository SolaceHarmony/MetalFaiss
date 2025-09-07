"""
test_binary_flat_index.py - Tests for binary flat index
"""

import unittest
import mlx.core as mx
from typing import List, Optional, Tuple

from ..index.binary_flat_index import BinaryFlatIndex
from ..vector_transform.binary_transform import BinaryRotationTransform
from ..types.metric_type import MetricType
from ..utils.search_result import SearchRangeResult

class TestBinaryFlatIndex(unittest.TestCase):
    """Test binary flat index."""
    
    def setUp(self):
        """Set up test data."""
        self.d = 64  # Must be multiple of 8
        self.n = 100
        self.x = mx.random.randint(0, 2, shape=(self.n, self.d), dtype=mx.uint8)
        
    def test_flat_index(self):
        """Test flat index basics."""
        index = BinaryFlatIndex(self.d)
        self.assertEqual(index.d, self.d)
        self.assertEqual(len(index), 0)
        
        # Add vectors
        index.add(self.x)
        self.assertEqual(len(index), self.n)
        
        # Test dimension validation
        with self.assertRaises(ValueError):
            BinaryFlatIndex(63)  # Not multiple of 8
            
    def test_search(self):
        """Test k-nearest neighbor search."""
        index = BinaryFlatIndex(self.d)
        index.add(self.x)
        
        # Search parameters
        k = 4
        nq = 10
        xq = self.x[:nq]
        
        # Search
        D, I = index.search(xq, k)
        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))
        
        # Verify distances are sorted
        self.assertTrue(mx.all(D[:, 1:] >= D[:, :-1]))
        
        # Verify exact match for query points
        self.assertTrue(mx.all(D[:, 0] == 0))
        self.assertTrue(mx.all(I[:, 0] == mx.arange(nq)))
        
    def test_range_search(self):
        """Test range search."""
        index = BinaryFlatIndex(self.d)
        index.add(self.x)
        
        # Search parameters
        radius = 10
        nq = 10
        xq = self.x[:nq]
        
        # Search
        result = index.range_search(xq, radius)
        self.assertIsInstance(result, SearchRangeResult)
        
        # Verify lims array
        self.assertEqual(len(result.lims), nq + 1)
        self.assertEqual(result.lims[0], 0)
        
        # Verify distances are within radius
        for d in result.distances:
            self.assertTrue(mx.all(d <= radius))
            
    def test_reconstruct(self):
        """Test vector reconstruction."""
        index = BinaryFlatIndex(self.d)
        index.add(self.x)
        
        # Single vector
        x_rec = index.reconstruct(0)
        self.assertTrue(mx.all(x_rec == self.x[0:1]))
        
        # Multiple vectors
        idx = mx.array([0, 2, 4])
        x_rec = index.reconstruct(idx)
        self.assertTrue(mx.all(x_rec == self.x[idx]))
        
        # Out of bounds
        with self.assertRaises(ValueError):
            index.reconstruct(self.n)
            
        with self.assertRaises(ValueError):
            index.reconstruct(mx.array([0, self.n]))
            
    def test_empty_index(self):
        """Test empty index behavior."""
        index = BinaryFlatIndex(self.d)
        
        # Search empty index
        D, I = index.search(self.x[:10], k=4)
        self.assertEqual(D.shape, (10, 4))
        self.assertEqual(I.shape, (10, 4))
        self.assertTrue(mx.all(I == 0))
        
        # Range search empty index
        result = index.range_search(self.x[:10], radius=10)
        self.assertEqual(len(result.distances), 0)
        self.assertEqual(len(result.indices), 0)
        self.assertTrue(mx.all(result.lims == 0))
        
    def test_transform(self):
        """Test with binary transform."""
        index = BinaryFlatIndex(self.d)
        transform = BinaryRotationTransform(self.d)
        transform.train(self.x)
        
        # Add transformed vectors
        y = transform.apply(self.x)
        index.add(y)
        
        # Search with transformed queries
        xq = self.x[:10]
        yq = transform.apply(xq)
        D, I = index.search(yq, k=4)
        
        # Verify exact matches
        self.assertTrue(mx.all(D[:, 0] == 0))
        self.assertTrue(mx.all(I[:, 0] == mx.arange(10)))

if __name__ == '__main__':
    unittest.main()
