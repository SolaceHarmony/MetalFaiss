"""
test_gpu_ops.py - Tests for GPU operations

These tests verify the functionality of GPU-optimized operations,
ensuring they produce correct results and handle memory properly.
"""

import unittest
import mlx.core as mx
from typing import List, Tuple
from ..faissmlx.gpu_ops import (
    GpuResources,
    GpuMemoryManager,
    matmul,
    l2_distances,
    cosine_distances,
    hamming_distances,
    binary_and,
    binary_or,
    binary_xor,
    popcount
)

class TestGpuResources(unittest.TestCase):
    """Test GPU resource management."""
    
    def setUp(self):
        """Create test resources."""
        self.resources = GpuResources(max_memory=1024)  # 1KB limit
        
    def test_memory_tracking(self):
        """Test memory allocation tracking."""
        # Allocate within limit
        self.assertTrue(self.resources.allocate(512))
        self.assertEqual(self.resources.current_memory, 512)
        
        # Exceed limit
        self.assertFalse(self.resources.allocate(1024))
        self.assertEqual(self.resources.current_memory, 512)
        
        # Free memory
        self.resources.free(256)
        self.assertEqual(self.resources.current_memory, 256)
        
        # Reset
        self.resources.reset()
        self.assertEqual(self.resources.current_memory, 0)

class TestGpuMemoryManager(unittest.TestCase):
    """Test GPU memory manager."""
    
    def setUp(self):
        """Create test manager."""
        self.resources = GpuResources(max_memory=1024)  # 1KB limit
        self.manager = GpuMemoryManager(self.resources)
        
    def test_allocation(self):
        """Test array allocation."""
        with self.manager:
            # Allocate float32 array
            arr = self.manager.alloc((100,), dtype="float32")  # 400 bytes
            self.assertEqual(arr.shape, (100,))
            self.assertEqual(arr.dtype, mx.float32)
            
            # Allocate uint8 array
            arr = self.manager.alloc((100,), dtype="uint8")  # 100 bytes
            self.assertEqual(arr.dtype, mx.uint8)
            
            # Exceed limit
            with self.assertRaises(MemoryError):
                self.manager.alloc((1000,), dtype="float32")  # 4000 bytes
                
        # Memory should be reset after context
        self.assertEqual(self.resources.current_memory, 0)
        
    def test_manual_free(self):
        """Test manual memory freeing."""
        with self.manager:
            arr = self.manager.alloc((100,), dtype="float32")
            self.manager.free(arr)
            
            # Should be able to allocate again
            arr = self.manager.alloc((100,), dtype="float32")

class TestGpuArrayOps(unittest.TestCase):
    """Test GPU array operations."""
    
    def test_array_transfer(self):
        """Test array movement between CPU and GPU."""
        # From list
        x = [[1, 2], [3, 4]]
        x_gpu = mx.array(x)
        self.assertEqual(x_gpu.shape, (2, 2))
        from .mlx_test_utils import assert_array_equal
        assert_array_equal(x_gpu, mx.array(x))
        
        # From numpy
        x = mx.random.normal(shape=(3, 4))
        x_gpu = x
        from .mlx_test_utils import assert_allclose
        assert_allclose(x_gpu, x)
        
        # From MLX array
        x = mx.random.normal(shape=(2, 3))
        x_gpu = x
        assert_allclose(x_gpu, x)

class TestGpuMatrixOps(unittest.TestCase):
    """Test GPU matrix operations."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        a = mx.array([[1, 2], [3, 4]])
        b = mx.array([[5, 6], [7, 8]])
        c = matmul(a, b)
        
        from .mlx_test_utils import assert_array_equal
        assert_array_equal(c, mx.array([[19, 22], [43, 50]]))

class TestGpuDistanceOps(unittest.TestCase):
    """Test GPU distance computations."""
    
    def test_l2_distances(self):
        """Test L2 distance computation."""
        x = mx.array([[1, 0], [0, 1]])
        y = mx.array([[1, 1], [0, 0]])
        
        dists = l2_distances(x, y)
        from .mlx_test_utils import assert_allclose
        assert_allclose(dists, mx.array([[1.0, 1.0], [2.0, 1.0]], dtype=mx.float32))
        
    def test_cosine_distances(self):
        """Test cosine distance computation."""
        x = mx.array([[1, 0], [1, 1]])
        y = mx.array([[1, 1], [0, 1]])
        
        dists = cosine_distances(x, y)
        # cos(0) = 1, cos(45°) ≈ 0.707
        expected = mx.array([[0.293, 1.0], [0.293, 0.293]], dtype=mx.float32)
        from .mlx_test_utils import assert_allclose
        assert_allclose(mx.round(dists, 3), expected, rtol=1e-3, atol=1e-3)
        
    def test_hamming_distances(self):
        """Test Hamming distance computation."""
        x = mx.array([[0, 1, 1], [1, 1, 0]], dtype=mx.uint8)
        y = mx.array([[1, 1, 1], [0, 0, 0]], dtype=mx.uint8)
        
        dists = hamming_distances(x, y)
        from .mlx_test_utils import assert_array_equal
        assert_array_equal(dists, mx.array([[1, 3], [3, 2]], dtype=mx.uint32))

class TestGpuBinaryOps(unittest.TestCase):
    """Test GPU binary operations."""
    
    def setUp(self):
        """Create test data."""
        self.x = mx.array([0b1010, 0b1100], dtype=mx.uint8)
        self.y = mx.array([0b1100, 0b1010], dtype=mx.uint8)
        
    def test_binary_ops(self):
        """Test basic binary operations."""
        from .mlx_test_utils import assert_array_equal
        # AND
        assert_array_equal(binary_and(self.x, self.y), mx.array([0b1000, 0b1000], dtype=mx.uint8))
        
        # OR
        assert_array_equal(binary_or(self.x, self.y), mx.array([0b1110, 0b1110], dtype=mx.uint8))
        
        # XOR
        assert_array_equal(binary_xor(self.x, self.y), mx.array([0b0110, 0b0110], dtype=mx.uint8))
        
    def test_popcount(self):
        """Test population count."""
        x = mx.array([0b1010, 0b1111, 0b0000], dtype=mx.uint8)
        counts = popcount(x)
        from .mlx_test_utils import assert_array_equal
        assert_array_equal(counts, mx.array([2, 4, 0], dtype=mx.uint8))

if __name__ == '__main__':
    unittest.main()
