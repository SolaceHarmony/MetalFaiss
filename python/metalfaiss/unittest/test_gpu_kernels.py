"""
test_gpu_kernels.py - Tests for GPU kernel implementations

These tests verify the functionality of GPU-optimized kernels,
ensuring they produce correct results and maintain expected performance
characteristics.
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Tuple
from ..faissmlx.gpu_kernels import (
    KernelConfig,
    batched_matmul,
    strided_matmul,
    l2_distance_kernel,
    cosine_distance_kernel,
    hamming_distance_kernel,
    binary_hamming_kernel,
    binary_and_kernel,
    binary_or_kernel,
    binary_xor_kernel,
    memcpy_kernel,
    memset_kernel,
    reduce_sum_kernel,
    scan_kernel
)

class TestKernelConfig(unittest.TestCase):
    """Test kernel configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KernelConfig()
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.max_shared_memory, 48 * 1024)
        self.assertEqual(config.warp_size, 32)
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = KernelConfig(
            block_size=128,
            max_shared_memory=32 * 1024,
            warp_size=16
        )
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.max_shared_memory, 32 * 1024)
        self.assertEqual(config.warp_size, 16)

class TestMatrixKernels(unittest.TestCase):
    """Test matrix operation kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
        # Create test matrices
        np.random.seed(42)
        self.a = mx.array(np.random.randn(3, 4))
        self.b = mx.array(np.random.randn(4, 5))
        
        # Batched matrices
        self.batch_a = mx.array(np.random.randn(2, 3, 4))
        self.batch_b = mx.array(np.random.randn(2, 4, 5))
        
    def test_batched_matmul(self):
        """Test batched matrix multiplication."""
        result = batched_matmul(self.batch_a, self.batch_b, self.config)
        
        # Compare with numpy batch matmul
        expected = np.matmul(
            self.batch_a.numpy(),
            self.batch_b.numpy()
        )
        np.testing.assert_array_almost_equal(
            result.numpy(),
            expected
        )
        
    def test_strided_matmul(self):
        """Test strided matrix multiplication."""
        stride = 2
        result = strided_matmul(self.a, self.b, stride, self.config)
        
        # Compare with numpy strided computation
        expected = np.matmul(
            self.a.numpy()[::stride],
            self.b.numpy()
        )
        np.testing.assert_array_almost_equal(
            result.numpy(),
            expected
        )

class TestDistanceKernels(unittest.TestCase):
    """Test distance computation kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
        # Create test vectors
        np.random.seed(42)
        self.x = mx.array(np.random.randn(10, 8))
        self.y = mx.array(np.random.randn(15, 8))
        
        # Binary vectors
        self.x_bin = mx.array(
            np.random.randint(0, 2, (10, 8)),
            dtype=mx.uint8
        )
        self.y_bin = mx.array(
            np.random.randint(0, 2, (15, 8)),
            dtype=mx.uint8
        )
        
    def test_l2_distance(self):
        """Test L2 distance computation."""
        result = l2_distance_kernel(self.x, self.y, self.config)
        
        # Compare with numpy computation
        x_np = self.x.numpy()
        y_np = self.y.numpy()
        expected = np.sum(x_np**2, axis=1)[:, None] + \
                  np.sum(y_np**2, axis=1) - \
                  2 * np.dot(x_np, y_np.T)
        np.testing.assert_array_almost_equal(
            result.numpy(),
            expected
        )
        
    def test_cosine_distance(self):
        """Test cosine distance computation."""
        result = cosine_distance_kernel(self.x, self.y, self.config)
        
        # Compare with numpy computation
        x_np = self.x.numpy()
        y_np = self.y.numpy()
        x_norm = np.sqrt(np.sum(x_np**2, axis=1))[:, None]
        y_norm = np.sqrt(np.sum(y_np**2, axis=1))[None, :]
        expected = 1 - np.dot(x_np, y_np.T) / (x_norm * y_norm)
        np.testing.assert_array_almost_equal(
            result.numpy(),
            expected
        )
        
    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        result = hamming_distance_kernel(
            self.x_bin,
            self.y_bin,
            self.config
        )
        
        # Compare with numpy computation
        x_np = self.x_bin.numpy()
        y_np = self.y_bin.numpy()
        expected = np.zeros((len(x_np), len(y_np)), dtype=np.uint32)
        for i in range(len(x_np)):
            for j in range(len(y_np)):
                expected[i, j] = np.sum(x_np[i] != y_np[j])
        np.testing.assert_array_equal(
            result.numpy(),
            expected
        )

class TestBinaryKernels(unittest.TestCase):
    """Test binary operation kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
        # Create binary vectors
        self.x = mx.array([0b1010, 0b1100], dtype=mx.uint8)
        self.y = mx.array([0b1100, 0b1010], dtype=mx.uint8)
        
    def test_binary_ops(self):
        """Test binary operations."""
        # AND
        result = binary_and_kernel(self.x, self.y, self.config)
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([0b1000, 0b1000], dtype=np.uint8)
        )
        
        # OR
        result = binary_or_kernel(self.x, self.y, self.config)
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([0b1110, 0b1110], dtype=np.uint8)
        )
        
        # XOR
        result = binary_xor_kernel(self.x, self.y, self.config)
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([0b0110, 0b0110], dtype=np.uint8)
        )
        
    def test_binary_hamming(self):
        """Test binary Hamming weight computation."""
        result = binary_hamming_kernel(self.x, self.y, self.config)
        
        # Compare with numpy computation
        x_np = self.x.numpy()
        y_np = self.y.numpy()
        expected = np.zeros((len(x_np), len(y_np)), dtype=np.uint32)
        for i in range(len(x_np)):
            for j in range(len(y_np)):
                expected[i, j] = bin(x_np[i] ^ y_np[j]).count('1')
        np.testing.assert_array_equal(
            result.numpy(),
            expected
        )

class TestMemoryKernels(unittest.TestCase):
    """Test memory management kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
    def test_memcpy(self):
        """Test memory copy."""
        src = mx.array([1, 2, 3, 4])
        dst = mx.zeros_like(src)
        
        memcpy_kernel(dst, src, self.config)
        np.testing.assert_array_equal(
            dst.numpy(),
            src.numpy()
        )
        
    def test_memset(self):
        """Test memory set."""
        arr = mx.zeros(5)
        value = 42
        
        memset_kernel(arr, value, self.config)
        np.testing.assert_array_equal(
            arr.numpy(),
            np.full(5, value)
        )

class TestUtilityKernels(unittest.TestCase):
    """Test utility kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
    def test_reduce_sum(self):
        """Test sum reduction."""
        x = mx.array([[1, 2, 3], [4, 5, 6]])
        
        # Total sum
        result = reduce_sum_kernel(x, None, self.config)
        self.assertEqual(float(result), 21)
        
        # Sum along axis
        result = reduce_sum_kernel(x, 0, self.config)
        np.testing.assert_array_equal(
            result.numpy(),
            [5, 7, 9]
        )
        
    def test_scan(self):
        """Test parallel scan."""
        x = mx.array([1, 2, 3, 4])
        
        # Inclusive scan
        result = scan_kernel(x, True, self.config)
        np.testing.assert_array_equal(
            result.numpy(),
            [1, 3, 6, 10]
        )
        
        # Exclusive scan
        result = scan_kernel(x, False, self.config)
        np.testing.assert_array_equal(
            result.numpy(),
            [0, 1, 3, 6]
        )

if __name__ == '__main__':
    unittest.main()