"""
test_mlx_gpu_core.py - Tests for core GPU operations

These tests verify low-level GPU operations like:
1. Selection algorithms (top-k, partitioning)
2. Distance computations
3. Vector operations
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Tuple

from ..faissmlx.ops import array
from ..faissmlx.gpu_ops import (
    GpuResources,
    GpuMemoryManager,
    to_gpu,
    from_gpu,
    gpu_l2_distances,
    gpu_cosine_distances,
    gpu_hamming_distances
)
from ..faissmlx.gpu_kernels import (
    KernelConfig,
    l2_distance_kernel,
    cosine_distance_kernel,
    hamming_distance_kernel
)

def make_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate test data."""
    np.random.seed(seed)
    return array(np.random.rand(num, d).astype('float32'))

def make_binary_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate binary test data."""
    np.random.seed(seed)
    return array(
        np.random.randint(0, 2, (num, d)).astype('uint8')
    )

class TestGpuSelect(unittest.TestCase):
    """Test GPU selection operations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.n = 10000
        self.k = 100
        self.resources = GpuResources()
        
        # Create test vectors
        self.x = make_data(self.n, self.d)
        
    def test_topk(self):
        """Test top-k selection."""
        # Create random distances
        distances = array(np.random.rand(self.n).astype('float32'))
        
        # Get top-k on CPU
        cpu_indices = np.argsort(distances.numpy())[:self.k]
        cpu_values = distances.numpy()[cpu_indices]
        
        # Get top-k on GPU
        gpu_distances = to_gpu(distances)
        gpu_values, gpu_indices = self.resources.topk(
            gpu_distances,
            self.k,
            largest=False
        )
        
        # Results should match
        np.testing.assert_array_equal(
            from_gpu(gpu_indices),
            cpu_indices
        )
        np.testing.assert_allclose(
            from_gpu(gpu_values),
            cpu_values,
            rtol=1e-5
        )
        
    def test_partition(self):
        """Test array partitioning."""
        # Create random values
        values = array(np.random.rand(self.n).astype('float32'))
        k = self.n // 2
        
        # Partition on CPU
        cpu_values = values.numpy()
        pivot = np.partition(cpu_values, k)[k]
        cpu_left = cpu_values[cpu_values <= pivot]
        cpu_right = cpu_values[cpu_values > pivot]
        
        # Partition on GPU
        gpu_values = to_gpu(values)
        gpu_pivot = self.resources.partition(gpu_values, k)
        gpu_values = from_gpu(gpu_values)
        
        # Check partition properties
        self.assertLessEqual(len(gpu_values[gpu_values <= gpu_pivot]), k)
        self.assertGreaterEqual(len(gpu_values[gpu_values > gpu_pivot]), self.n - k)

class TestGpuDistance(unittest.TestCase):
    """Test GPU distance computations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.resources = GpuResources()
        
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
        cpu_distances = np.zeros((self.nq, self.nb), dtype=np.float32)
        for i in range(self.nq):
            for j in range(self.nb):
                cpu_distances[i, j] = np.sum(
                    (self.xq[i].numpy() - self.xb[j].numpy()) ** 2
                )
                
        # GPU implementations
        gpu_distances = gpu_l2_distances(self.xq, self.xb)
        kernel_distances = l2_distance_kernel(
            self.xq,
            self.xb,
            KernelConfig()
        )
        
        # All should match
        np.testing.assert_allclose(
            gpu_distances.numpy(),
            cpu_distances,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            kernel_distances.numpy(),
            cpu_distances,
            rtol=1e-5
        )
        
    def test_cosine_distance(self):
        """Test cosine distance computation."""
        # CPU implementation
        cpu_distances = np.zeros((self.nq, self.nb), dtype=np.float32)
        for i in range(self.nq):
            q = self.xq[i].numpy()
            q_norm = np.sqrt(np.sum(q * q))
            for j in range(self.nb):
                b = self.xb[j].numpy()
                b_norm = np.sqrt(np.sum(b * b))
                cpu_distances[i, j] = 1 - np.sum(q * b) / (q_norm * b_norm)
                
        # GPU implementations
        gpu_distances = gpu_cosine_distances(self.xq, self.xb)
        kernel_distances = cosine_distance_kernel(
            self.xq,
            self.xb,
            KernelConfig()
        )
        
        # All should match
        np.testing.assert_allclose(
            gpu_distances.numpy(),
            cpu_distances,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            kernel_distances.numpy(),
            cpu_distances,
            rtol=1e-5
        )
        
    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        # CPU implementation
        cpu_distances = np.zeros((self.nq, self.nb), dtype=np.uint32)
        for i in range(self.nq):
            for j in range(self.nb):
                cpu_distances[i, j] = np.sum(
                    self.xq_bin[i].numpy() != self.xb_bin[j].numpy()
                )
                
        # GPU implementations
        gpu_distances = gpu_hamming_distances(self.xq_bin, self.xb_bin)
        kernel_distances = hamming_distance_kernel(
            self.xq_bin,
            self.xb_bin,
            KernelConfig()
        )
        
        # All should match
        np.testing.assert_array_equal(
            gpu_distances.numpy(),
            cpu_distances
        )
        np.testing.assert_array_equal(
            kernel_distances.numpy(),
            cpu_distances
        )

class TestGpuVectorOps(unittest.TestCase):
    """Test GPU vector operations."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.n = 1000
        self.resources = GpuResources()
        
        # Create test vectors
        self.x = make_data(self.n, self.d)
        
    def test_reduction(self):
        """Test reduction operations."""
        # Sum reduction
        cpu_sum = np.sum(self.x.numpy())
        gpu_sum = self.resources.sum(to_gpu(self.x))
        np.testing.assert_allclose(
            float(gpu_sum),
            cpu_sum,
            rtol=1e-5
        )
        
        # Mean reduction
        cpu_mean = np.mean(self.x.numpy())
        gpu_mean = self.resources.mean(to_gpu(self.x))
        np.testing.assert_allclose(
            float(gpu_mean),
            cpu_mean,
            rtol=1e-5
        )
        
    def test_elementwise(self):
        """Test elementwise operations."""
        y = make_data(self.n, self.d)
        
        # Addition
        cpu_add = self.x.numpy() + y.numpy()
        gpu_add = self.resources.add(
            to_gpu(self.x),
            to_gpu(y)
        )
        np.testing.assert_allclose(
            from_gpu(gpu_add),
            cpu_add,
            rtol=1e-5
        )
        
        # Multiplication
        cpu_mul = self.x.numpy() * y.numpy()
        gpu_mul = self.resources.multiply(
            to_gpu(self.x),
            to_gpu(y)
        )
        np.testing.assert_allclose(
            from_gpu(gpu_mul),
            cpu_mul,
            rtol=1e-5
        )

if __name__ == '__main__':
    unittest.main()