"""
test_mlx_gpu_core.py - Tests for core GPU operations

These tests verify low-level GPU operations like:
1. Selection algorithms (top-k, partitioning)
2. Distance computations
3. Vector operations
"""

import unittest
import mlx.core as mx
from typing import List, Tuple

from ..faissmlx.ops import array
from ..faissmlx.resources import Resources, MemoryManager
from ..faissmlx.ops import (
    l2_distances,
    cosine_distances,
    hamming_distances
)
from ..faissmlx.gpu_kernels import (
    KernelConfig,
    l2_distance_kernel,
    cosine_distance_kernel,
    hamming_distance_kernel
)

from ..utils.rng_utils import new_key, split2

def make_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate test data using MLX RNG keys."""
    k = new_key(seed)
    return mx.random.uniform(shape=(num, d), key=k).astype(mx.float32)

def make_binary_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate binary test data (uint8) using MLX RNG keys."""
    k = new_key(seed)
    u = mx.random.uniform(shape=(num, d), key=k)
    return (u >= mx.array(0.5, dtype=u.dtype)).astype(mx.uint8)

class TestGpuSelect(unittest.TestCase):
    """Test GPU selection operations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.n = 10000
        self.k = 100
        self.resources = Resources()
        
        # Create test vectors
        self.x = make_data(self.n, self.d)
        
    def test_topk(self):
        """Test top-k selection."""
        # Create random distances
        k = new_key(123)
        distances = mx.random.uniform(shape=(self.n,), key=k).astype(mx.float32)
        
        # Get top-k on CPU
        cpu_indices = mx.argsort(distances)[: self.k]
        cpu_values = mx.take(distances, cpu_indices)
        
        # Get top-k on GPU
        # Direct MLX arrays (GPU-only project)
        gpu_values, gpu_indices = self.resources.topk(
            distances,
            self.k,
            largest=False
        )
        # Results should match
        self.assertTrue(bool(mx.all(mx.equal(gpu_indices, cpu_indices)).item()))
        self.assertTrue(bool(mx.allclose(gpu_values, cpu_values, rtol=1e-5, atol=1e-7).item()))
        
    def test_partition(self):
        """Test array partitioning."""
        # Create random values
        k = new_key(321)
        values = mx.random.uniform(shape=(self.n,), key=k).astype(mx.float32)
        k = self.n // 2
        
        # CPU reference via argsort gather
        order = mx.argsort(values)
        pivot = values[order[k]]
        
        # Partition on GPU
        gpu_pivot = self.resources.partition(values, k)
        
        # Check partition properties
        left_count = int(mx.sum(mx.less_equal(values, mx.array(float(gpu_pivot), dtype=values.dtype))).item())
        right_count = int(mx.sum(mx.greater(values, mx.array(float(gpu_pivot), dtype=values.dtype))).item())
        self.assertLessEqual(left_count, k)
        self.assertGreaterEqual(right_count, self.n - k)

class TestGpuDistance(unittest.TestCase):
    """Test GPU distance computations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.resources = Resources()
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
        # Create binary vectors
        self.d_bin = 64
        self.xb_bin = make_binary_data(self.nb, self.d_bin)
        self.xq_bin = make_binary_data(self.nq, self.d_bin)
        
    def test_l2_distance(self):
        """Test L2 distance computation."""
        # CPU implementation
        diffs = self.xq[:, None, :] - self.xb[None, :, :]
        cpu_distances = mx.sum(diffs * diffs, axis=2)
                
        # GPU implementations
        gpu_distances = l2_distances(self.xq, self.xb)
        kernel_distances = l2_distance_kernel(
            self.xq,
            self.xb,
            KernelConfig()
        )
        
        # All should match
        self.assertTrue(bool(mx.allclose(gpu_distances, cpu_distances, rtol=1e-5, atol=1e-7).item()))
        self.assertTrue(bool(mx.allclose(kernel_distances, cpu_distances, rtol=1e-5, atol=1e-7).item()))
        
    def test_cosine_distance(self):
        """Test cosine distance computation."""
        # CPU implementation
        q_norms = mx.sqrt(mx.sum(self.xq * self.xq, axis=1, keepdims=True))
        b_norms = mx.sqrt(mx.sum(self.xb * self.xb, axis=1, keepdims=True))
        dots = mx.matmul(self.xq, mx.transpose(self.xb))
        norms = mx.matmul(q_norms, mx.transpose(b_norms))
        cpu_distances = mx.subtract(mx.ones_like(dots), mx.divide(dots, norms))
                
        # GPU implementations
        gpu_distances = cosine_distances(self.xq, self.xb)
        kernel_distances = cosine_distance_kernel(
            self.xq,
            self.xb,
            KernelConfig()
        )
        
        # All should match
        self.assertTrue(bool(mx.allclose(gpu_distances, cpu_distances, rtol=1e-5, atol=1e-7).item()))
        self.assertTrue(bool(mx.allclose(kernel_distances, cpu_distances, rtol=1e-5, atol=1e-7).item()))
        
    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        # CPU implementation
        cpu_distances = mx.sum(mx.not_equal(self.xq_bin[:, None, :], self.xb_bin[None, :, :]).astype(mx.uint32), axis=2)
                
        # GPU implementations
        gpu_distances = hamming_distances(self.xq_bin, self.xb_bin)
        kernel_distances = hamming_distance_kernel(
            self.xq_bin,
            self.xb_bin,
            KernelConfig()
        )
        
        # All should match
        self.assertTrue(bool(mx.all(mx.equal(gpu_distances, cpu_distances)).item()))
        self.assertTrue(bool(mx.all(mx.equal(kernel_distances, cpu_distances)).item()))

class TestGpuVectorOps(unittest.TestCase):
    """Test GPU vector operations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.n = 1000
        self.resources = Resources()
        
        # Create test vectors
        self.x = make_data(self.n, self.d)
        
    def test_reduction(self):
        """Test reduction operations."""
        # Sum reduction
        cpu_sum = float(mx.sum(self.x).item())
        gpu_sum = self.resources.sum(self.x)
        self.assertAlmostEqual(float(gpu_sum), cpu_sum, places=5)
        
        # Mean reduction
        cpu_mean = float(mx.mean(self.x).item())
        gpu_mean = self.resources.mean(self.x)
        self.assertAlmostEqual(float(gpu_mean), cpu_mean, places=5)
        
    def test_elementwise(self):
        """Test elementwise operations."""
        y = make_data(self.n, self.d)
        
        # Addition
        cpu_add = mx.add(self.x, y)
        gpu_add = self.resources.add(self.x, y)
        self.assertTrue(bool(mx.allclose(gpu_add, cpu_add, rtol=1e-5, atol=1e-7).item()))
        
        # Multiplication
        cpu_mul = mx.multiply(self.x, y)
        gpu_mul = self.resources.multiply(self.x, y)
        self.assertTrue(bool(mx.allclose(gpu_mul, cpu_mul, rtol=1e-5, atol=1e-7).item()))

if __name__ == '__main__':
    unittest.main()
