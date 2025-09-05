"""
test_gpu_ops.py - Tests for GPU operations

These tests verify the functionality of GPU-optimized operations,
ensuring they produce correct results and handle memory properly.
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Tuple
from ..faissmlx.gpu_ops import (
    GpuResources,
    GpuMemoryManager,
    to_gpu,
    from_gpu,
    gpu_matmul,
    gpu_l2_distances,
    gpu_cosine_distances,
    gpu_hamming_distances,
    gpu_binary_and,
    gpu_binary_or,
    gpu_binary_xor,
    gpu_popcount
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
        x_gpu = to_gpu(x)
        self.assertEqual(x_gpu.shape, (2, 2))
        np.testing.assert_array_equal(
            from_gpu(x_gpu),
            np.array(x)
        )
        
        # From numpy
        x = np.random.randn(3, 4)
        x_gpu = to_gpu(x)
        np.testing.assert_array_almost_equal(
            from_gpu(x_gpu),
            x
        )
        
        # From MLX array
        x = mx.random.normal((2, 3))
        x_gpu = to_gpu(x)
        np.testing.assert_array_almost_equal(
            from_gpu(x_gpu),
            x.numpy()
        )

class TestGpuMatrixOps(unittest.TestCase):
    """Test GPU matrix operations."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        a = to_gpu([[1, 2], [3, 4]])
        b = to_gpu([[5, 6], [7, 8]])
        c = gpu_matmul(a, b)
        
        np.testing.assert_array_equal(
            from_gpu(c),
            np.array([[19, 22], [43, 50]])
        )

class TestGpuDistanceOps(unittest.TestCase):
    """Test GPU distance computations."""
    
    def test_l2_distances(self):
        """Test L2 distance computation."""
        x = to_gpu([[1, 0], [0, 1]])
        y = to_gpu([[1, 1], [0, 0]])
        
        dists = gpu_l2_distances(x, y)
        np.testing.assert_array_almost_equal(
            from_gpu(dists),
            np.array([[1, 1], [2, 1]])
        )
        
    def test_cosine_distances(self):
        """Test cosine distance computation."""
        x = to_gpu([[1, 0], [1, 1]])
        y = to_gpu([[1, 1], [0, 1]])
        
        dists = gpu_cosine_distances(x, y)
        # cos(0) = 1, cos(45°) ≈ 0.707
        expected = np.array([[0.293, 1], [0.293, 0.293]])
        np.testing.assert_array_almost_equal(
            from_gpu(dists),
            expected,
            decimal=3
        )
        
    def test_hamming_distances(self):
        """Test Hamming distance computation."""
        x = to_gpu([[0, 1, 1], [1, 1, 0]], dtype="uint8")
        y = to_gpu([[1, 1, 1], [0, 0, 0]], dtype="uint8")
        
        dists = gpu_hamming_distances(x, y)
        np.testing.assert_array_equal(
            from_gpu(dists),
            np.array([[1, 3], [3, 2]])
        )

class TestGpuBinaryOps(unittest.TestCase):
    """Test GPU binary operations."""
    
    def setUp(self):
        """Create test data."""
        self.x = to_gpu([0b1010, 0b1100], dtype="uint8")
        self.y = to_gpu([0b1100, 0b1010], dtype="uint8")
        
    def test_binary_ops(self):
        """Test basic binary operations."""
        # AND
        np.testing.assert_array_equal(
            from_gpu(gpu_binary_and(self.x, self.y)),
            np.array([0b1000, 0b1000], dtype=np.uint8)
        )
        
        # OR
        np.testing.assert_array_equal(
            from_gpu(gpu_binary_or(self.x, self.y)),
            np.array([0b1110, 0b1110], dtype=np.uint8)
        )
        
        # XOR
        np.testing.assert_array_equal(
            from_gpu(gpu_binary_xor(self.x, self.y)),
            np.array([0b0110, 0b0110], dtype=np.uint8)
        )
        
    def test_popcount(self):
        """Test population count."""
        x = to_gpu([0b1010, 0b1111, 0b0000], dtype="uint8")
        counts = gpu_popcount(x)
        np.testing.assert_array_equal(
            from_gpu(counts),
            np.array([2, 4, 0], dtype=np.uint8)
        )

if __name__ == '__main__':
    unittest.main()