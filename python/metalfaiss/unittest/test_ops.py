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
        self.assertEqual(arr.tolist(), data)
        
        # With dtype
        arr = array(data, dtype="float32")
        self.assertEqual(arr.dtype, mx.float32)
        
        # From numpy
        data = mx.random.normal(shape=(3, 4))
        arr = array(data.tolist())
        self.assertEqual(arr.tolist(), data.tolist())
        
    def test_zeros_ones(self):
        """Test zeros and ones creation."""
        shape = (2, 3)
        
        # Zeros
        z = zeros(shape)
        self.assertEqual(z.shape, shape)
        self.assertEqual(z.tolist(), [[0, 0, 0], [0, 0, 0]])
        
        # Ones
        o = ones(shape)
        self.assertEqual(o.shape, shape)
        self.assertEqual(o.tolist(), [[1, 1, 1], [1, 1, 1]])
        
    def test_arange(self):
        """Test arange."""
        # Basic range
        arr = arange(5)
        self.assertEqual(arr.tolist(), [0, 1, 2, 3, 4])
        
        # With start, stop
        arr = arange(2, 5)
        self.assertEqual(arr.tolist(), [2, 3, 4])
        
        # With step
        arr = arange(0, 6, 2)
        self.assertEqual(arr.tolist(), [0, 2, 4])
        
    def test_concatenate(self):
        """Test array concatenation."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6]])
        
        # Along axis 0
        c = concatenate([a, b])
        self.assertEqual(c.tolist(), [[1, 2], [3, 4], [5, 6]])
        
        # Along axis 1
        a = array([[1, 2], [3, 4]])
        b = array([[5], [6]])
        c = concatenate([a, b], axis=1)
        self.assertEqual(c.tolist(), [[1, 2, 5], [3, 4, 6]])

class TestMathOps(unittest.TestCase):
    """Test mathematical operations."""
    
    def setUp(self):
        """Create test data."""
        self.x = array([[1, 2, 3], [4, 5, 6]])
        
    def test_reductions(self):
        """Test reduction operations."""
        # Sum
        self.assertEqual(float(sum(self.x)), 21)  # Total sum
        self.assertEqual(sum(self.x, axis=0).tolist(), [5, 7, 9])
        self.assertEqual(sum(self.x, axis=1).tolist(), [6, 15])
        
        # Mean
        self.assertEqual(float(mean(self.x)), 3.5)
        self.assertEqual([round(v, 6) for v in mean(self.x, axis=0).tolist()], [2.5, 3.5, 4.5])
        
        # Min/Max
        self.assertEqual(float(min(self.x)), 1)
        self.assertEqual(float(max(self.x)), 6)
        self.assertEqual(min(self.x, axis=0).tolist(), [1, 2, 3])
        self.assertEqual(max(self.x, axis=1).tolist(), [3, 6])

class TestMatrixOps(unittest.TestCase):
    """Test matrix operations."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        c = matmul(a, b)
        self.assertEqual(c.tolist(), [[19, 22], [43, 50]])
        
    def test_transpose(self):
        """Test transpose."""
        x = array([[1, 2, 3], [4, 5, 6]])
        y = transpose(x)
        self.assertEqual(y.tolist(), [[1, 4], [2, 5], [3, 6]])
        
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
        self.assertEqual(dists.tolist(), [[1.0, 1.0], [2.0, 1.0]])
        
    def test_cosine_distances(self):
        """Test cosine distance computation."""
        x = array([[1, 0], [1, 1]])
        y = array([[1, 1], [0, 1]])
        
        dists = cosine_distances(x, y)
        # cos(0) = 1, cos(45°) ≈ 0.707
        expected = [[0.293, 1], [0.293, 0.293]]
        got = [[round(v, 3) for v in row] for row in dists.tolist()]
        self.assertEqual(got, expected)
        
    def test_hamming_distances(self):
        """Test Hamming distance computation."""
        x = array([[0, 1, 1], [1, 1, 0]], dtype="uint8")
        y = array([[1, 1, 1], [0, 0, 0]], dtype="uint8")
        
        dists = hamming_distances(x, y)
        self.assertEqual(dists.tolist(), [[1, 3], [3, 2]])

class TestBinaryOps(unittest.TestCase):
    """Test binary operations."""
    
    def setUp(self):
        """Create test data."""
        self.x = array([0b1010, 0b1100], dtype="uint8")
        self.y = array([0b1100, 0b1010], dtype="uint8")
        
    def test_binary_ops(self):
        """Test basic binary operations."""
        # AND
        self.assertEqual(binary_and(self.x, self.y).tolist(), [0b1000, 0b1000])
        
        # OR
        self.assertEqual(binary_or(self.x, self.y).tolist(), [0b1110, 0b1110])
        
        # XOR
        self.assertEqual(binary_xor(self.x, self.y).tolist(), [0b0110, 0b0110])
        
        # NOT
        self.assertEqual(binary_not(array([0b0011], dtype="uint8")).tolist(), [0b11111100])
        
    def test_popcount(self):
        """Test population count."""
        x = array([0b1010, 0b1111, 0b0000], dtype="uint8")
        counts = popcount(x)
        self.assertEqual(counts.tolist(), [2, 4, 0])

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
        self.assertEqual(x.tolist(), y.tolist())

if __name__ == '__main__':
    unittest.main()
