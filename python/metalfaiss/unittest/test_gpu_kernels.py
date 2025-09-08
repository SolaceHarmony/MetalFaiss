"""
test_gpu_kernels.py - Tests for GPU kernel implementations

These tests verify the functionality of GPU-optimized kernels,
ensuring they produce correct results and maintain expected performance
characteristics.
"""

import unittest
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

from .mlx_test_utils import assert_allclose, assert_array_equal, randn, randint


class TestMatrixKernels(unittest.TestCase):
    """Test matrix operation kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
        # Create test matrices
        self.a = mx.random.normal(shape=(3, 4))
        self.b = mx.random.normal(shape=(4, 5))
        # Batched matrices
        self.batch_a = mx.random.normal(shape=(2, 3, 4))
        self.batch_b = mx.random.normal(shape=(2, 4, 5))
        
    def test_batched_matmul(self):
        """Test batched matrix multiplication."""
        result = batched_matmul(self.batch_a, self.batch_b, self.config)
        
        # Compare with MLX matmul
        expected = mx.matmul(self.batch_a, self.batch_b)
        assert_allclose(result, expected)
        
    def test_strided_matmul(self):
        """Test strided matrix multiplication."""
        stride = 2
        result = strided_matmul(self.a, self.b, stride, self.config)
        
        # Compare with MLX strided computation
        expected = mx.matmul(self.a[::stride], self.b)
        assert_allclose(result, expected)

class TestDistanceKernels(unittest.TestCase):
    """Test distance computation kernels."""
    
    def setUp(self):
        """Create test data."""
        self.config = KernelConfig()
        
        # Create test vectors
        self.x = mx.random.normal(shape=(10, 8))
        self.y = mx.random.normal(shape=(15, 8))
        # Binary vectors
        self.x_bin = mx.random.randint(0, 2, shape=(10, 8), dtype=mx.uint8)
        self.y_bin = mx.random.randint(0, 2, shape=(15, 8), dtype=mx.uint8)
        
    def test_l2_distance(self):
        """Test L2 distance computation."""
        result = l2_distance_kernel(self.x, self.y, self.config)
        
        # Compare with MLX computation
        x_norms = mx.sum(self.x * self.x, axis=1, keepdims=True)
        y_norms = mx.sum(self.y * self.y, axis=1)
        expected = x_norms + y_norms[None, :] - 2 * mx.matmul(self.x, self.y.T)
        assert_allclose(result, expected)
        
    def test_cosine_distance(self):
        """Test cosine distance computation."""
        result = cosine_distance_kernel(self.x, self.y, self.config)
        
        # Compare with MLX computation
        x_norm = mx.sqrt(mx.sum(self.x * self.x, axis=1, keepdims=True))
        y_norm = mx.sqrt(mx.sum(self.y * self.y, axis=1, keepdims=True)).T
        expected = 1.0 - mx.matmul(self.x, self.y.T) / (x_norm * y_norm)
        assert_allclose(result, expected)
        
    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        result = hamming_distance_kernel(
            self.x_bin,
            self.y_bin,
            self.config
        )
        
        # Compare with MLX computation
        expected = mx.zeros((self.x_bin.shape[0], self.y_bin.shape[0]), dtype=mx.uint32)
        for i in range(int(self.x_bin.shape[0])):
            xi = self.x_bin[i]
            # broadcast compare and sum across axis 1
            diffs = mx.not_equal(xi[None, :], self.y_bin)  # (nb, d)
            counts = mx.sum(diffs, axis=1)
            expected[:, i] = counts  # fill column-wise
        assert_array_equal(result, expected.T)

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
        assert_array_equal(result, mx.array([0b1000, 0b1000], dtype=mx.uint8))
        
        # OR
        result = binary_or_kernel(self.x, self.y, self.config)
        assert_array_equal(result, mx.array([0b1110, 0b1110], dtype=mx.uint8))
        
        # XOR
        result = binary_xor_kernel(self.x, self.y, self.config)
        assert_array_equal(result, mx.array([0b0110, 0b0110], dtype=mx.uint8))
        
    def test_binary_hamming(self):
        """Test binary Hamming weight computation."""
        result = binary_hamming_kernel(self.x, self.y, self.config)
        
        # Compare with MLX computation
        expected = mx.zeros((self.x.shape[0], self.y.shape[0]), dtype=mx.uint32)
        for i in range(int(self.x.shape[0])):
            for j in range(int(self.y.shape[0])):
                expected[i, j] = int(bin(int(self.x[i].item()) ^ int(self.y[j].item())).count('1'))  # boundary-ok
        assert_array_equal(result, expected)

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
        assert_array_equal(dst, src)
        
    def test_memset(self):
        """Test memory set."""
        arr = mx.zeros(5)
        value = 42
        
        memset_kernel(arr, value, self.config)
        assert_array_equal(arr, mx.full((5,), value))

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
        self.assertEqual(float(result), 21.0)
        
        # Sum along axis
        result = reduce_sum_kernel(x, 0, self.config)
        assert_array_equal(result, mx.array([5, 7, 9]))
        
    def test_scan(self):
        """Test parallel scan."""
        x = mx.array([1, 2, 3, 4])
        
        # Inclusive scan
        result = scan_kernel(x, True, self.config)
        assert_array_equal(result, mx.array([1, 3, 6, 10]))
        
        # Exclusive scan
        result = scan_kernel(x, False, self.config)
        assert_array_equal(result, mx.array([0, 1, 3, 6]))

if __name__ == '__main__':
    unittest.main()
