"""
test_mlx_basics.py - Basic MLX functionality tests

These tests verify fundamental MLX operations and ensure compatibility
with the original FAISS GPU implementation patterns.
"""

import unittest
import mlx.core as mx
from typing import List, Tuple
from ..faissmlx.ops import (
    Device,
    array,
    zeros,
    ones,
    arange,
    concatenate,
    matmul,
    l2_distances,
    cosine_distances,
    hamming_distances
)
from ..faissmlx.gpu_ops import (
    GpuResources,
    GpuMemoryManager,
    to_gpu,
    from_gpu,
    gpu_matmul,
    gpu_l2_distances,
    gpu_cosine_distances,
    gpu_hamming_distances
)
from ..faissmlx.gpu_kernels import (
    KernelConfig,
    batched_matmul,
    l2_distance_kernel,
    cosine_distance_kernel,
    hamming_distance_kernel
)

from .mlx_test_utils import assert_array_equal, assert_allclose


def make_data(num: int, d: int) -> mx.array:
    """Generate test data using MLX."""
    return mx.random.uniform(shape=(num, d))

def make_binary_data(num: int, d: int) -> mx.array:
    """Generate binary test data using MLX."""
    return mx.random.randint(0, 2, shape=(num, d), dtype=mx.uint8)

class TestMLXBasics(unittest.TestCase):
    """Test basic MLX functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.nb = 1000
        self.nq = 100
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
        # Binary vectors
        self.d_bin = 64
        self.xb_bin = make_binary_data(self.nb, self.d_bin)
        self.xq_bin = make_binary_data(self.nq, self.d_bin)
        
    def test_array_ops(self):
        """Test basic array operations."""
        # Creation
        x = array([[1, 2], [3, 4]])
        self.assertEqual(x.shape, (2, 2))
        
        # Basic ops
        z = zeros((2, 3))
        self.assertEqual(z.shape, (2, 3))
        self.assertTrue(mx.all(z == 0))
        
        o = ones((2, 3))
        self.assertTrue(mx.all(o == 1))
        
        # Concatenation
        c = concatenate([x, x])
        self.assertEqual(c.shape, (4, 2))
        
    def test_matmul(self):
        """Test matrix multiplication."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        
        # CPU
        c = matmul(a, b)
        assert_array_equal(c, mx.array([[19, 22], [43, 50]]))
        
        # GPU
        c_gpu = gpu_matmul(a, b)
        assert_array_equal(c_gpu, c)
        
        # Batched
        batch_a = array([a.tolist(), a.tolist()])
        batch_b = array([b.tolist(), b.tolist()])
        c_batch = batched_matmul(batch_a, batch_b)
        self.assertEqual(c_batch.shape, (2, 2, 2))

class TestDistanceMetrics(unittest.TestCase):
    """Test distance computations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.nb = 1000
        self.nq = 100
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
    def test_l2_distance(self):
        """Test L2 distance computation."""
        # CPU implementation
        dist_cpu = l2_distances(self.xq[:10], self.xb[:10])
        
        # GPU implementation
        dist_gpu = gpu_l2_distances(self.xq[:10], self.xb[:10])
        
        # Kernel implementation
        dist_kernel = l2_distance_kernel(
            self.xq[:10],
            self.xb[:10],
            KernelConfig()
        )
        
        # All should match
        assert_allclose(dist_cpu, dist_gpu)
        assert_allclose(dist_cpu, dist_kernel)
        
    def test_cosine_distance(self):
        """Test cosine distance computation."""
        # CPU implementation
        dist_cpu = cosine_distances(self.xq[:10], self.xb[:10])
        
        # GPU implementation
        dist_gpu = gpu_cosine_distances(self.xq[:10], self.xb[:10])
        
        # Kernel implementation
        dist_kernel = cosine_distance_kernel(
            self.xq[:10],
            self.xb[:10],
            KernelConfig()
        )
        
        # All should match
        assert_allclose(dist_cpu, dist_gpu)
        assert_allclose(dist_cpu, dist_kernel)

class TestBinaryOperations(unittest.TestCase):
    """Test binary vector operations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64  # Multiple of 8 for binary
        self.nb = 1000
        self.nq = 100
        
        # Create binary vectors
        self.xb = make_binary_data(self.nb, self.d)
        self.xq = make_binary_data(self.nq, self.d)
        
    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        # CPU implementation
        dist_cpu = hamming_distances(self.xq[:10], self.xb[:10])
        
        # GPU implementation
        dist_gpu = gpu_hamming_distances(self.xq[:10], self.xb[:10])
        
        # Kernel implementation
        dist_kernel = hamming_distance_kernel(
            self.xq[:10],
            self.xb[:10],
            KernelConfig()
        )
        
        # All should match
        assert_array_equal(dist_cpu, dist_gpu)
        assert_array_equal(dist_cpu, dist_kernel)
        
        # Verify against numpy
        for i in range(10):
            for j in range(10):
                expected = np.sum(
                    self.xq[i].numpy() != self.xb[j].numpy()
                )
                self.assertEqual(
                    dist_cpu[i, j],
                    expected
                )

class TestMemoryManagement(unittest.TestCase):
    """Test GPU memory management."""
    
    def setUp(self):
        """Create resources."""
        self.resources = GpuResources(max_memory=1024 * 1024)  # 1MB limit
        self.manager = GpuMemoryManager(self.resources)
        
    def test_allocation(self):
        """Test memory allocation."""
        with self.manager:
            # Allocate within limit
            arr = self.manager.alloc((1000,), dtype="float32")
            self.assertEqual(arr.shape, (1000,))
            
            # Exceed limit
            with self.assertRaises(MemoryError):
                self.manager.alloc((1000000,), dtype="float32")
                
    def test_device_transfer(self):
        """Test device transfers."""
        x = array([1, 2, 3, 4])
        
        # To GPU
        x_gpu = to_gpu(x)
        assert_array_equal(x_gpu, x)
        
        # Back to CPU
        x_cpu = from_gpu(x_gpu)
        assert_array_equal(x_cpu, x)

if __name__ == '__main__':
    unittest.main()
