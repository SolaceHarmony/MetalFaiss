"""
test_transforms.py - Tests for vector transforms
"""

import unittest
import mlx.core as mx
from typing import List, Optional, Tuple

from ..vector_transform import (
    BaseVectorTransform,
    RandomRotationTransform,
    PCAMatrixTransform,
    ITQTransform,
    OPQTransform,
    RemapDimensionsTransform,
    NormalizationTransform,
    CenteringTransform
)

class TestTransforms(unittest.TestCase):
    """Test vector transforms."""
    
    def setUp(self):
        """Set up test data."""
        self.d = 64
        self.n = 100
        # MLX-only random input
        self.x = mx.random.normal(shape=(self.n, self.d)).astype(mx.float32)
        
    def test_random_rotation(self):
        """Test random rotation transform."""
        transform = RandomRotationTransform(self.d)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test matrix properties
        self.assertEqual(transform.rotation_matrix.shape, (self.d, self.d))
        
        # Test orthogonality
        R = transform.rotation_matrix
        RtR = mx.matmul(mx.transpose(R), R)
        self.assertTrue(bool(mx.allclose(RtR, mx.eye(self.d), rtol=1e-5, atol=1e-5)))
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        self.assertTrue(bool(mx.allclose(x_rec, self.x, rtol=1e-5, atol=1e-5)))
        
    def test_pca_matrix(self):
        """Test PCA matrix transform."""
        transform = PCAMatrixTransform(self.d)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test matrix properties
        self.assertEqual(transform.pca_matrix.shape, (self.d, self.d))
        
        # Test orthogonality
        P = transform.pca_matrix
        PtP = mx.matmul(mx.transpose(P), P)
        self.assertTrue(bool(mx.allclose(PtP, mx.eye(self.d), rtol=1e-5, atol=1e-5)))
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        self.assertTrue(bool(mx.allclose(x_rec, self.x, rtol=1e-5, atol=1e-5)))
        
    def test_itq(self):
        """Test ITQ transform."""
        transform = ITQTransform(self.d)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test matrix properties
        self.assertEqual(transform.rotation_matrix.shape, (self.d, self.d))
        
        # Test orthogonality
        R = transform.rotation_matrix
        RtR = mx.matmul(mx.transpose(R), R)
        self.assertTrue(bool(mx.allclose(RtR, mx.eye(self.d), rtol=1e-5, atol=1e-5)))
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        self.assertTrue(bool(mx.allclose(x_rec, self.x, rtol=1e-5, atol=1e-5)))
        
    def test_opq(self):
        """Test OPQ transform."""
        M = 8  # Number of sub-quantizers
        transform = OPQTransform(self.d, M)
        self.assertEqual(transform.d_in, self.d)
        self.assertEqual(transform.d_out, self.d)
        self.assertEqual(transform.M, M)
        
        # Test training
        transform.train(self.x)
        self.assertTrue(transform.is_trained)
        
        # Test matrix properties
        self.assertEqual(transform.rotation_matrix.shape, (self.d, self.d))
        
        # Test orthogonality
        R = transform.rotation_matrix
        RtR = mx.matmul(mx.transpose(R), R)
        self.assertTrue(bool(mx.allclose(RtR, mx.eye(self.d), rtol=1e-5, atol=1e-5)))
        
        # Test forward transform
        y = transform.apply(self.x)
        self.assertEqual(y.shape, self.x.shape)
        
        # Test inverse transform
        x_rec = transform.reverse_transform(y)
        self.assertTrue(bool(mx.allclose(x_rec, self.x, rtol=1e-5, atol=1e-5)))
        
    def test_simple_transforms(self):
        """Test simple transforms."""
        # Test remap dimensions
        indices = [0, 2, 4, 6]
        transform = RemapDimensionsTransform(self.d, indices)
        y = transform.apply(self.x)
        self.assertEqual(y.shape, (self.n, len(indices)))
        self.assertTrue(bool(mx.all(mx.equal(y, self.x[:, indices]))))
        
        # Test normalization
        transform = NormalizationTransform(self.d)
        y = transform.apply(self.x)
        norms = mx.sqrt(mx.sum(y * y, axis=1))
        self.assertTrue(bool(mx.allclose(norms, mx.ones_like(norms), rtol=1e-6, atol=1e-6)))
        
        # Test centering
        transform = CenteringTransform(self.d)
        transform.train(self.x)
        y = transform.apply(self.x)
        means = mx.mean(y, axis=0)
        self.assertTrue(bool(mx.allclose(means, mx.zeros_like(means), rtol=1e-6, atol=1e-6)))
        
    def test_dimension_error(self):
        """Test error with wrong dimensions."""
        transform = RandomRotationTransform(self.d)
        transform.train(self.x)
        
        # Wrong input dimension
        x_wrong = mx.random.normal(shape=(10, self.d + 1)).astype(mx.float32)
        with self.assertRaises(ValueError):
            transform.apply(x_wrong)
            
        # Wrong output dimension
        y_wrong = mx.random.normal(shape=(10, self.d + 1)).astype(mx.float32)
        with self.assertRaises(ValueError):
            transform.reverse_transform(y_wrong)

if __name__ == '__main__':
    unittest.main()
