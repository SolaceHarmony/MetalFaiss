"""
test_ops.py - Tests for MLX operations

These tests verify the functionality of our MLX operation wrappers,
ensuring they behave correctly and match expected numerical results.
"""

import unittest
import mlx.core as mx
from typing import List, Tuple
from ..faissmlx.ops import (
    Device,
    array, zeros, ones, arange, concatenate,
    sum, mean, min, max,
    matmul, transpose,
    l2_distances, cosine_distances, hamming_distances,
    binary_and, binary_or, binary_xor, binary_not, popcount,
    to_device, get_device
)

class TestArrayOps(unittest.TestCase):
    """Test array creation and manipulation ops."""
    
    def test_array_creation(self):
        """Test array creation."""
        # From list
        data = [[1, 2], [3, 4]]
        arr = array(data)
        self.assertTrue(bool(mx.all(mx.equal(arr, mx.array(data))).item()))  # boundary-ok
        
        # With dtype
        arr = array(data, dtype="float32")
        self.assertEqual(arr.dtype, mx.float32)
        
        # From numpy
        data = mx.random.normal(shape=(3, 4))
        arr = array(data)
        self.assertTrue(bool(mx.all(mx.equal(arr, data)).item()))  # boundary-ok
        
    def test_zeros_ones(self):
        """Test zeros and ones creation."""
        shape = (2, 3)
        
        # Zeros
        z = zeros(shape)
        self.assertEqual(z.shape, shape)
        self.assertTrue(bool(mx.all(mx.equal(z, mx.zeros(shape))).item()))  # boundary-ok
        
        # Ones
        o = ones(shape)
        self.assertEqual(o.shape, shape)
        self.assertTrue(bool(mx.all(mx.equal(o, mx.ones(shape))).item()))  # boundary-ok
        
    def test_arange(self):
        """Test arange."""
        # Basic range
        arr = arange(5)
        self.assertTrue(bool(mx.all(mx.equal(arr, mx.arange(5))).item()))  # boundary-ok
        
        # With start, stop
        arr = arange(2, 5)
        self.assertTrue(bool(mx.all(mx.equal(arr, mx.arange(2, 5))).item()))  # boundary-ok
        
        # With step
        arr = arange(0, 6, 2)
        self.assertTrue(bool(mx.all(mx.equal(arr, mx.array([0, 2, 4]))).item()))  # boundary-ok
        
    def test_concatenate(self):
        """Test array concatenation."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6]])
        
        # Along axis 0
        c = concatenate([a, b])
        self.assertTrue(bool(mx.all(mx.equal(c, mx.array([[1,2],[3,4],[5,6]]))).item()))  # boundary-ok
        
        # Along axis 1
        a = array([[1, 2], [3, 4]])
        b = array([[5], [6]])
        c = concatenate([a, b], axis=1)
        self.assertTrue(bool(mx.all(mx.equal(c, mx.array([[1,2,5],[3,4,6]]))).item()))  # boundary-ok

class TestMathOps(unittest.TestCase):
    """Test mathematical operations."""
    
    def setUp(self):
        """Create test data."""
        self.x = array([[1, 2, 3], [4, 5, 6]])
        
    def test_reductions(self):
        """Test reduction operations."""
        # Sum
        self.assertEqual(float(sum(self.x)), 21)  # Total sum
        self.assertTrue(bool(mx.all(mx.equal(sum(self.x, axis=0), mx.array([5,7,9]))).item()))  # boundary-ok
        self.assertTrue(bool(mx.all(mx.equal(sum(self.x, axis=1), mx.array([6,15]))).item()))  # boundary-ok
        
        # Mean
        self.assertEqual(float(mean(self.x)), 3.5)
        self.assertTrue(bool(mx.allclose(mean(self.x, axis=0), mx.array([2.5,3.5,4.5]), rtol=1e-6, atol=1e-6).item()))  # boundary-ok
        
        # Min/Max
        self.assertEqual(float(min(self.x)), 1)
        self.assertEqual(float(max(self.x)), 6)
        self.assertTrue(bool(mx.all(mx.equal(min(self.x, axis=0), mx.array([1,2,3]))).item()))  # boundary-ok
        self.assertTrue(bool(mx.all(mx.equal(max(self.x, axis=1), mx.array([3,6]))).item()))  # boundary-ok

class TestMatrixOps(unittest.TestCase):
    """Test matrix operations."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        c = matmul(a, b)
        self.assertTrue(bool(mx.all(mx.equal(c, mx.array([[19,22],[43,50]]))).item()))  # boundary-ok
        
    def test_transpose(self):
        """Test transpose."""
        x = array([[1, 2, 3], [4, 5, 6]])
        y = transpose(x)
        self.assertTrue(bool(mx.all(mx.equal(y, mx.array([[1,4],[2,5],[3,6]]))).item()))  # boundary-ok
        
        # With custom axes
        x = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        y = transpose(x, (1, 0, 2))
        self.assertEqual(y.shape, (2, 2, 2))

class TestDistanceOps(unittest.TestCase):
    """Test distance computations."""
    
    def test_l2_distances(self):
        """Test L2 distance computation."""
        x = array([[1, 0], [0, 1]])
        y = array([[1, 1], [0, 0]])
        
        dists = l2_distances(x, y)
        self.assertTrue(bool(mx.allclose(dists, mx.array([[1.0,1.0],[2.0,1.0]]), rtol=1e-6, atol=1e-6).item()))  # boundary-ok
        
    def test_cosine_distances(self):
        """Test cosine distance computation."""
        x = array([[1, 0], [1, 1]])
        y = array([[1, 1], [0, 1]])
        
        dists = cosine_distances(x, y)
        expected = mx.array([[0.293, 1.0],[0.293, 0.293]], dtype=mx.float32)
        self.assertTrue(bool(mx.allclose(mx.round(dists, 3), expected, rtol=1e-3, atol=1e-3).item()))  # boundary-ok
        
    def test_hamming_distances(self):
        """Test Hamming distance computation."""
        x = array([[0, 1, 1], [1, 1, 0]], dtype="uint8")
        y = array([[1, 1, 1], [0, 0, 0]], dtype="uint8")
        
        dists = hamming_distances(x, y)
        self.assertTrue(bool(mx.all(mx.equal(dists, mx.array([[1,3],[3,2]], dtype=mx.int32))).item()))  # boundary-ok

class TestBinaryOps(unittest.TestCase):
    """Test binary operations."""
    
    def setUp(self):
        """Create test data."""
        self.x = array([0b1010, 0b1100], dtype="uint8")
        self.y = array([0b1100, 0b1010], dtype="uint8")
        
    def test_binary_ops(self):
        """Test basic binary operations."""
        # AND
        self.assertTrue(bool(mx.all(mx.equal(binary_and(self.x, self.y), mx.array([0b1000,0b1000], dtype='uint8'))).item()))  # boundary-ok
        
        # OR
        self.assertTrue(bool(mx.all(mx.equal(binary_or(self.x, self.y), mx.array([0b1110,0b1110], dtype='uint8'))).item()))  # boundary-ok
        
        # XOR
        self.assertTrue(bool(mx.all(mx.equal(binary_xor(self.x, self.y), mx.array([0b0110,0b0110], dtype='uint8'))).item()))  # boundary-ok
        
        # NOT
        self.assertTrue(bool(mx.all(mx.equal(binary_not(array([0b0011], dtype="uint8")), mx.array([0b11111100], dtype='uint8'))).item()))  # boundary-ok
        
    def test_popcount(self):
        """Test population count."""
        x = array([0b1010, 0b1111, 0b0000], dtype="uint8")
        counts = popcount(x)
        self.assertTrue(bool(mx.all(mx.equal(counts, mx.array([2,4,0], dtype='uint8'))).item()))  # boundary-ok

class TestDeviceOps(unittest.TestCase):
    """Test device operations."""
    
    def test_device_ops(self):
        """Test device placement and queries."""
        x = array([1, 2, 3])
        
        # For now, everything is on CPU
        self.assertEqual(get_device(x), Device.CPU)
        
        # Moving to device is no-op for now
        y = to_device(x, Device.GPU)
        self.assertEqual(get_device(y), Device.CPU)
        self.assertTrue(bool(mx.all(mx.equal(x, y)).item()))  # boundary-ok

if __name__ == '__main__':
    unittest.main()
