"""
test_mlx_index.py - Tests for MLX-based indices

These tests verify index functionality using MLX, particularly focusing on:
1. IVF search with preassigned lists
2. Coarse quantizer integration
3. Index transfer between CPU and GPU
4. Index ID handling
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Tuple
from ..faissmlx.ops import (
    Device,
    array,
    matmul,
    l2_distances
)
from ..faissmlx.gpu_ops import (
    GpuResources,
    GpuMemoryManager,
    to_gpu,
    from_gpu
)
from ..index.flat_index import FlatIndex
from ..index.ivf_flat_index import IVFFlatIndex
from ..index.binary_flat_index import BinaryFlatIndex
from ..index.binary_ivf_index import BinaryIVFIndex

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

class TestIVFSearch(unittest.TestCase):
    """Test IVF search functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64
        self.nb = 10000
        self.nq = 100
        self.nlist = 100
        self.k = 10
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
    def test_preassigned_search(self):
        """Test search with preassigned lists."""
        # Create and train index
        index = IVFFlatIndex(self.d, self.nlist)
        index.train(self.xb)
        index.add(self.xb)
        
        # Get coarse assignments
        quantizer = FlatIndex(self.d)
        quantizer.add(index.centroids)
        assign_d, assign_i = quantizer.search(self.xq, index.nprobe)
        
        # Search with assignments
        d1, i1 = index.search_preassigned(
            self.xq,
            self.k,
            assign_i,
            assign_d
        )
        
        # Regular search
        d2, i2 = index.search(self.xq, self.k)
        
        # Results should match
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_allclose(d1, d2, rtol=1e-5)
        
    def test_coarse_quantizer(self):
        """Test coarse quantizer integration."""
        # Create separate coarse quantizer
        quantizer = FlatIndex(self.d)
        
        # Create index with this quantizer
        index = IVFFlatIndex(self.d, self.nlist, quantizer=quantizer)
        
        # Train and add
        index.train(self.xb)
        index.add(self.xb)
        
        # Search
        d, i = index.search(self.xq, self.k)
        
        # Verify results
        self.assertEqual(d.shape, (self.nq, self.k))
        self.assertEqual(i.shape, (self.nq, self.k))

class TestBinaryIVFSearch(unittest.TestCase):
    """Test binary IVF search functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64  # Multiple of 8 for binary
        self.nb = 10000
        self.nq = 100
        self.nlist = 100
        self.k = 10
        
        # Create binary vectors
        self.xb = make_binary_data(self.nb, self.d)
        self.xq = make_binary_data(self.nq, self.d)
        
    def test_preassigned_search(self):
        """Test search with preassigned lists."""
        # Create and train index
        index = BinaryIVFIndex(self.d, self.nlist)
        index.train(self.xb)
        index.add(self.xb)
        
        # Get coarse assignments
        quantizer = BinaryFlatIndex(self.d)
        quantizer.add(index.centroids)
        assign_d, assign_i = quantizer.search(self.xq, index.nprobe)
        
        # Search with assignments
        d1, i1 = index.search_preassigned(
            self.xq,
            self.k,
            assign_i,
            assign_d
        )
        
        # Regular search
        d2, i2 = index.search(self.xq, self.k)
        
        # Results should match
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_array_equal(d1, d2)
        
    def test_coarse_quantizer(self):
        """Test coarse quantizer integration."""
        # Create separate coarse quantizer
        quantizer = BinaryFlatIndex(self.d)
        
        # Create index with this quantizer
        index = BinaryIVFIndex(self.d, self.nlist, quantizer=quantizer)
        
        # Train and add
        index.train(self.xb)
        index.add(self.xb)
        
        # Search
        d, i = index.search(self.xq, self.k)
        
        # Verify results
        self.assertEqual(d.shape, (self.nq, self.k))
        self.assertEqual(i.shape, (self.nq, self.k))

class TestIndexTransfer(unittest.TestCase):
    """Test index transfer between CPU and GPU."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64
        self.nb = 1000
        self.nq = 10
        self.k = 5
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
    def test_flat_transfer(self):
        """Test flat index transfer."""
        # Create CPU index
        index_cpu = FlatIndex(self.d)
        index_cpu.add(self.xb)
        
        # Search on CPU
        d_cpu, i_cpu = index_cpu.search(self.xq, self.k)
        
        # Transfer to GPU
        resources = GpuResources()
        index_gpu = index_cpu.to_gpu(resources)
        
        # Search on GPU
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should match
        np.testing.assert_array_equal(i_cpu, i_gpu)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-5)
        
    def test_ivf_transfer(self):
        """Test IVF index transfer."""
        # Create CPU index
        index_cpu = IVFFlatIndex(self.d, 10)
        index_cpu.train(self.xb)
        index_cpu.add(self.xb)
        
        # Search on CPU
        d_cpu, i_cpu = index_cpu.search(self.xq, self.k)
        
        # Transfer to GPU
        resources = GpuResources()
        index_gpu = index_cpu.to_gpu(resources)
        
        # Search on GPU
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should be similar (approximate search)
        self.assertGreaterEqual(
            (i_cpu == i_gpu).sum(),
            i_cpu.size * 0.8  # 80% match
        )

class TestIndexIDs(unittest.TestCase):
    """Test index ID handling."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64
        self.nb = 1000
        self.nq = 10
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.ids = np.arange(self.nb, dtype=np.int64)
        
        # Add large offset to test 64-bit handling
        self.ids_64bit = self.ids + (1 << 32)
        
    def test_flat_ids(self):
        """Test flat index ID handling."""
        index = FlatIndex(self.d)
        
        # Add with 64-bit IDs
        index.add_with_ids(self.xb, self.ids_64bit)
        
        # Search
        d, i = index.search(self.xb[:10], 1)
        
        # Should find exact matches with correct IDs
        np.testing.assert_array_equal(
            i[:, 0],
            self.ids_64bit[:10]
        )
        
    def test_ivf_ids(self):
        """Test IVF index ID handling."""
        index = IVFFlatIndex(self.d, 10)
        index.train(self.xb)
        
        # Add with 64-bit IDs
        index.add_with_ids(self.xb, self.ids_64bit)
        
        # Search
        d, i = index.search(self.xb[:10], 1)
        
        # Should find exact matches with correct IDs
        np.testing.assert_array_equal(
            i[:, 0],
            self.ids_64bit[:10]
        )

if __name__ == '__main__':
    unittest.main()