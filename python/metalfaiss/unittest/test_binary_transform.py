"""
test_binary_transform.py - Tests for binary vector transforms
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Optional, Tuple

from ..vector_transform.binary_transform import (
    BaseBinaryTransform,
    BinaryRotationTransform,
    BinaryMatrixTransform
)

class TestBinaryTransform(unittest.TestCase):
    """Test binary vector transforms."""
    
    def setUp(self):
        """Set up test data."""
        self.d = 64  # Must be multiple of 8
        self.n = 100
        np.random.seed(42)
        self.x = mx.array(
            np.random.randint(0, 2, (self.n, self.d)),
            dtype=np.uint8
        )
        
    def test_base_transform(self):
        """Test base binary transform."""
        transform = BaseBinaryTransform(self.d)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        
        # Test dimension validation
        with self.assertRaises(ValueError):
            BaseBinaryTransform(63)  # Not multiple of 8
            
        with self.assertRaises(ValueError):
            BaseBinaryTransform(64, d_out=63)  # Not multiple of 8
            
    def test_rotation_transform(self):
        """Test binary rotation transform."""
        transform = BinaryRotationTransform(self.d)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test permutation
        self.assertEqual(len(transform.permutation), self.d)
        self.assertTrue(all(i in transform.permutation for i in range(self.d)))
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(y.dtype, self.x.dtype)
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        np.testing.assert_array_equal(x_rec, self.x)
        
    def test_matrix_transform(self):
        """Test binary matrix transform."""
        transform = BinaryMatrixTransform(self.d)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test matrix shape
        self.assertEqual(transform.matrix.shape, (self.d, self.d))
        self.assertEqual(transform.matrix.dtype, np.uint8)
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(y.dtype, self.x.dtype)
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        self.assertEqual(x_rec.shape, self.x.shape)
        self.assertEqual(x_rec.dtype, self.x.dtype)
        
    def test_dimension_reduction(self):
        """Test dimension reduction."""
        d_out = 32  # Must be multiple of 8
        transform = BinaryMatrixTransform(self.d, d_out)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, d_out)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test matrix shape
        self.assertEqual(transform.matrix.shape, (self.d, d_out))
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, (self.n, d_out))
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        self.assertEqual(x_rec.shape, self.x.shape)
        
    def test_untrained_error(self):
        """Test error when using untrained transform."""
        transform = BinaryRotationTransform(self.d)
        with self.assertRaises(RuntimeError):
            transform.apply(self.x)
            
        transform = BinaryMatrixTransform(self.d)
        with self.assertRaises(RuntimeError):
            transform.apply(self.x)
            
    def test_dimension_error(self):
        """Test error with wrong dimensions."""
        transform = BinaryRotationTransform(self.d)
        transform.train(self.x)
        
        # Wrong input dimension
        x_wrong = mx.array(np.random.randint(0, 2, (10, self.d + 8)))
        with self.assertRaises(ValueError):
            transform.apply(x_wrong)
            
        # Wrong output dimension
        y_wrong = mx.array(np.random.randint(0, 2, (10, self.d + 8)))
        with self.assertRaises(ValueError):
            transform.reverse_transform(y_wrong)

if __name__ == '__main__':
    unittest.main()