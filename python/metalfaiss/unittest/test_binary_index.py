"""
test_binary_index.py - Tests for binary indices
"""

import unittest
import mlx.core as mx
from typing import List, Optional, Tuple

from ..index.binary_index import BaseBinaryIndex
from ..vector_transform.binary_transform import BinaryRotationTransform
from ..types.metric_type import MetricType

class TestBinaryIndex(unittest.TestCase):
    """Test binary index base class."""
    
    def setUp(self):
        """Set up test data."""
        self.d = 64  # Must be multiple of 8
        self.n = 100
        self.x = mx.random.randint(0, 2, shape=(self.n, self.d), dtype=mx.uint8)
        
    def test_base_index(self):
        """Test base binary index."""
        index = BaseBinaryIndex(self.d)
        self.assertEqual(index.d, self.d)
        self.assertEqual(len(index), 0)
        
        # Test dimension validation
        with self.assertRaises(ValueError):
            BaseBinaryIndex(63)  # Not multiple of 8
            
    def test_transform(self):
        """Test binary transform."""
        index = BaseBinaryIndex(self.d)
        transform = BinaryRotationTransform(self.d)
        transform.train(self.x)
        
        # Apply transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(y.dtype, self.x.dtype)
        
        # Verify transform preserves Hamming distances
        for i in range(10):
            for j in range(i + 1, 10):
                d_orig = mx.sum(self.x[i] != self.x[j])
                d_trans = mx.sum(y[i] != y[j])
                self.assertEqual(d_orig, d_trans)
                
    def test_search_not_implemented(self):
        """Test search methods raise NotImplementedError."""
        index = BaseBinaryIndex(self.d)
        
        with self.assertRaises(NotImplementedError):
            index.search(self.x, k=4)
            
        with self.assertRaises(NotImplementedError):
            index.range_search(self.x, radius=10)
            
    def test_reconstruct_not_implemented(self):
        """Test reconstruct method raises NotImplementedError."""
        index = BaseBinaryIndex(self.d)
        
        with self.assertRaises(NotImplementedError):
            index.reconstruct(0)
            
        with self.assertRaises(NotImplementedError):
            index.reconstruct(mx.array([0, 1, 2]))
            
    def test_reset(self):
        """Test reset method."""
        index = BaseBinaryIndex(self.d)
        self.assertEqual(len(index), 0)
        
        # Add some vectors
        index.ntotal = 100
        self.assertEqual(len(index), 100)
        
        # Reset
        index.reset()
        self.assertEqual(len(index), 0)

if __name__ == '__main__':
    unittest.main()
