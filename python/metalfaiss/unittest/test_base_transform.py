"""
test_base_transform.py - Tests for base vector transform classes

These tests verify the functionality of BaseVectorTransform and BaseLinearTransform,
which form the foundation for all vector transforms in MetalFaiss.
"""

import unittest
import mlx.core as mx
from ..vector_transform.base_vector_transform import (
    BaseVectorTransform,
    BaseLinearTransform
)

class MockVectorTransform(BaseVectorTransform):
    """Mock transform for testing base class."""
    
    def apply_noalloc(self, x: mx.array, xt: mx.array) -> None:
        """Mock implementation that copies/truncates input to output width."""
        xt[:] = x[:, : xt.shape[1]]
        
    def check_identical(self, other: 'BaseVectorTransform') -> None:
        """Mock implementation that checks dimensions."""
        super().check_identical(other)
        if not isinstance(other, MockVectorTransform):
            raise ValueError("Not a mock transform")

class MockLinearTransform(BaseLinearTransform):
    """Mock linear transform for testing."""
    
    def __init__(self, d_in: int, d_out: int, have_bias: bool = False):
        """Initialize with identity matrix."""
        super().__init__(d_in, d_out, have_bias)
        self.A = mx.eye(d_out, d_in)
        if have_bias:
            self.b = mx.zeros(d_out)
        self._is_trained = True
        self.is_orthonormal = True

class TestBaseVectorTransform(unittest.TestCase):
    """Test BaseVectorTransform functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d_in = 8
        self.d_out = 4
        self.transform = MockVectorTransform(self.d_in, self.d_out)
        
        # Create random test vectors (MLX)
        self.test_vectors = mx.random.normal(shape=(10, self.d_in)).astype(mx.float32)
        
    def test_initialization(self):
        """Test transform initialization."""
        self.assertEqual(self.transform.d_in, self.d_in)
        self.assertEqual(self.transform.d_out, self.d_out)
        self.assertFalse(self.transform.is_trained)
        
    def test_dimensions(self):
        """Test dimension validation."""
        # Invalid input dimension
        with self.assertRaises(ValueError):
            self.transform.apply([
                [0] * (self.d_in + 1)  # Wrong input dimension
            ])
            
    def test_training_required(self):
        """Test training requirement."""
        # Cannot apply untrained transform
        with self.assertRaises(RuntimeError):
            self.transform.apply(self.test_vectors.tolist())
            
        # Train the transform
        self.transform._is_trained = True
        
        # Now application should work
        result = self.transform.apply(self.test_vectors.tolist())
        self.assertEqual(result.shape, (self.test_vectors.shape[0], self.d_out))
        
    def test_reverse_transform(self):
        """Test reverse transform."""
        # Base class should raise NotImplementedError
        self.transform._is_trained = True
        with self.assertRaises(NotImplementedError):
            self.transform.reverse_transform([[0] * self.d_out])
            
    def test_check_identical(self):
        """Test identity checking."""
        # Different dimensions
        other = MockVectorTransform(self.d_in + 1, self.d_out)
        with self.assertRaises(ValueError):
            self.transform.check_identical(other)
            
        # Different training status
        other = MockVectorTransform(self.d_in, self.d_out)
        self.transform._is_trained = True
        with self.assertRaises(ValueError):
            self.transform.check_identical(other)
            
        # Different types
        other = BaseVectorTransform(self.d_in, self.d_out)
        with self.assertRaises(ValueError):
            self.transform.check_identical(other)

class TestBaseLinearTransform(unittest.TestCase):
    """Test BaseLinearTransform functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d_in = 8
        self.d_out = 4
        self.transform = MockLinearTransform(self.d_in, self.d_out)
        
        # Create random test vectors (MLX)
        self.test_vectors = mx.random.normal(shape=(10, self.d_in)).astype(mx.float32)
        
    def test_initialization(self):
        """Test transform initialization."""
        # Without bias
        transform = MockLinearTransform(self.d_in, self.d_out)
        self.assertFalse(transform.have_bias)
        self.assertIsNone(transform.b)
        
        # With bias
        transform = MockLinearTransform(self.d_in, self.d_out, have_bias=True)
        self.assertTrue(transform.have_bias)
        self.assertIsNotNone(transform.b)
        
    def test_linear_transform(self):
        """Test linear transformation."""
        # Identity transform should return input
        result = self.transform.apply(self.test_vectors.tolist())
        self.assertTrue(bool(mx.allclose(result[:, :self.d_out], self.test_vectors[:, :self.d_out], rtol=1e-6, atol=1e-6)))
        
        # With bias
        transform = MockLinearTransform(self.d_in, self.d_out, have_bias=True)
        transform.b = mx.ones(self.d_out)
        result = transform.apply(self.test_vectors.tolist())
        self.assertTrue(bool(mx.allclose(result, self.test_vectors[:, :self.d_out] + 1, rtol=1e-6, atol=1e-6)))
        
    def test_transform_transpose(self):
        """Test transpose transform."""
        x = mx.array(self.test_vectors)
        y = mx.zeros((len(x), self.d_out))
        
        # Apply forward transform
        self.transform.apply_noalloc(x, y)
        
        # Apply transpose transform
        x_recovered = mx.zeros((len(x), self.d_in))
        self.transform.transform_transpose(y, x_recovered)
        
        # For orthonormal matrix, should recover input
        self.assertTrue(bool(mx.allclose(x_recovered[:, :self.d_out], x[:, :self.d_out], rtol=1e-6, atol=1e-6)))
        
    def test_reverse_transform(self):
        """Test reverse transform."""
        # Only works for orthonormal matrices
        self.transform.is_orthonormal = False
        with self.assertRaises(ValueError):
            self.transform.reverse_transform([[0] * self.d_out])
            
        # Should work when orthonormal
        self.transform.is_orthonormal = True
        result = self.transform.reverse_transform(self.test_vectors[:, :self.d_out].tolist())
        self.assertTrue(bool(mx.allclose(result[:, :self.d_out], self.test_vectors[:, :self.d_out], rtol=1e-6, atol=1e-6)))
        
    def test_set_is_orthonormal(self):
        """Test orthonormality checking."""
        # Identity matrix should be orthonormal
        self.transform.set_is_orthonormal()
        self.assertTrue(self.transform.is_orthonormal)
        
        # Non-orthonormal matrix
        self.transform.A = mx.array([
            [1, 1],
            [1, 1]
        ])
        self.transform.set_is_orthonormal()
        self.assertFalse(self.transform.is_orthonormal)
        
    def test_check_identical(self):
        """Test identity checking."""
        # Different bias settings
        other = MockLinearTransform(self.d_in, self.d_out, have_bias=True)
        with self.assertRaises(ValueError):
            self.transform.check_identical(other)
            
        # Different matrices
        other = MockLinearTransform(self.d_in, self.d_out)
        other.A = mx.zeros_like(other.A)
        with self.assertRaises(ValueError):
            self.transform.check_identical(other)
            
        # Different types
        other = BaseLinearTransform(self.d_in, self.d_out)
        with self.assertRaises(ValueError):
            self.transform.check_identical(other)

if __name__ == '__main__':
    unittest.main()
