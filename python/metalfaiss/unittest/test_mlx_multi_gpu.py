"""
test_mlx_multi_gpu.py - Tests for multi-GPU support

These tests verify that indices can be properly sharded across multiple
GPUs while maintaining accuracy and performance.
"""

import unittest
import mlx.core as mx
from typing import List, Tuple
import time

from ..faissmlx.ops import array
from ..utils.rng_utils import new_key
from ..faissmlx.resources import Resources
from ..index.flat_index import FlatIndex
from ..index.ivf_flat_index import IVFFlatIndex
from ..index.product_quantizer_index import ProductQuantizerIndex
from ..index.scalar_quantizer_index import ScalarQuantizerIndex
from ..index.binary_flat_index import BinaryFlatIndex
from ..index.binary_ivf_index import BinaryIVFIndex

def make_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate test data."""
    k = new_key(seed)
    return mx.random.uniform(shape=(num, d), key=k).astype(mx.float32)

def make_binary_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate binary test data."""
    k = new_key(seed)
    # Draw uniform [0,1) then threshold at 0.5 to get bits
    u = mx.random.uniform(shape=(num, d), key=k)
    return (u >= mx.array(0.5, dtype=u.dtype)).astype(mx.uint8)

class TestShardedFlat(unittest.TestCase):
    """Test sharded flat index."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.nb = 1000
        self.nq = 200
        self.k = 10
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
    def test_sharded_search(self):
        """Test search with sharded index."""
        # Create CPU index for reference
        index_cpu = FlatIndex(self.d)
        index_cpu.add(self.xb)
        d_ref, i_ref = index_cpu.search(self.xq, self.k)
        
        # Create sharded GPU index
        resources = [Resources() for _ in range(2)]
        index_gpu = index_cpu.to_gpus(resources, shard=True)
        
        # Search
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should match
        self.assertTrue(bool(mx.all(mx.equal(i_gpu, i_ref)).item()))
        self.assertTrue(bool(mx.allclose(d_gpu, d_ref, rtol=1e-5, atol=1e-7).item()))
        
        # Should not be able to add to sharded index
        with self.assertRaises(RuntimeError):
            index_gpu.add(self.xq)

class TestShardedIVF(unittest.TestCase):
    """Test sharded IVF indices."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.nb = 10000
        self.nq = 200
        self.nlist = 100
        self.k = 10
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
    def test_sharded_ivf_flat(self):
        """Test sharded IVF flat index."""
        # Create and train CPU index
        index_cpu = IVFFlatIndex(self.d, self.nlist)
        index_cpu.train(self.xb)
        index_cpu.add(self.xb)
        
        # Set search parameters
        index_cpu.nprobe = 8
        d_ref, i_ref = index_cpu.search(self.xq, self.k)
        
        # Create sharded GPU index with shared quantizer
        resources = [Resources() for _ in range(2)]
        index_gpu = index_cpu.to_gpus(
            resources,
            shard=True,
            share_quantizer=True
        )
        
        # Set same parameters
        index_gpu.nprobe = 8
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should be very close
        match_ratio = (i_gpu == i_ref).sum() / i_ref.size
        self.assertGreater(match_ratio, 0.95)
        
        # Reset and add again
        index_gpu.reset()
        index_gpu.add(self.xb)
        
        d_gpu2, i_gpu2 = index_gpu.search(self.xq, self.k)
        match_ratio = (i_gpu2 == i_ref).sum() / i_ref.size
        self.assertGreater(match_ratio, 0.95)
        
    def test_sharded_ivf_pq(self):
        """Test sharded IVF PQ index."""
        # Create and train CPU index
        index_cpu = ProductQuantizerIndex(self.d, self.nlist, M=4)
        index_cpu.train(self.xb)
        index_cpu.add(self.xb)
        
        # Set search parameters
        index_cpu.nprobe = 8
        d_ref, i_ref = index_cpu.search(self.xq, self.k)
        
        # Create sharded GPU index
        resources = [Resources() for _ in range(2)]
        index_gpu = index_cpu.to_gpus(
            resources,
            shard=True,
            share_quantizer=True
        )
        
        # Set same parameters
        index_gpu.nprobe = 8
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should be similar (PQ is lossy)
        match_ratio = (i_gpu == i_ref).sum() / i_ref.size
        self.assertGreater(match_ratio, 0.8)

class TestShardedBinary(unittest.TestCase):
    """Test sharded binary indices."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64  # Multiple of 8 for binary
        self.nb = 10000
        self.nq = 200
        self.nlist = 100
        self.k = 10
        
        # Create binary vectors
        self.xb = make_binary_data(self.nb, self.d)
        self.xq = make_binary_data(self.nq, self.d)
        
    def test_sharded_binary_flat(self):
        """Test sharded binary flat index."""
        # Create CPU index
        index_cpu = BinaryFlatIndex(self.d)
        index_cpu.add(self.xb)
        d_ref, i_ref = index_cpu.search(self.xq, self.k)
        
        # Create sharded GPU index
        resources = [Resources() for _ in range(2)]
        index_gpu = index_cpu.to_gpus(resources, shard=True)
        
        # Search
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should match exactly
        self.assertTrue(bool(mx.all(mx.equal(i_gpu, i_ref)).item()))
        self.assertTrue(bool(mx.all(mx.equal(d_gpu, d_ref)).item()))
        
    def test_sharded_binary_ivf(self):
        """Test sharded binary IVF index."""
        # Create and train CPU index
        index_cpu = BinaryIVFIndex(self.d, self.nlist)
        index_cpu.train(self.xb)
        index_cpu.add(self.xb)
        
        # Set search parameters
        index_cpu.nprobe = 8
        d_ref, i_ref = index_cpu.search(self.xq, self.k)
        
        # Create sharded GPU index
        resources = [Resources() for _ in range(2)]
        index_gpu = index_cpu.to_gpus(
            resources,
            shard=True,
            share_quantizer=True
        )
        
        # Set same parameters
        index_gpu.nprobe = 8
        d_gpu, i_gpu = index_gpu.search(self.xq, self.k)
        
        # Results should be very close
        match_ratio = (i_gpu == i_ref).sum() / i_ref.size
        self.assertGreater(match_ratio, 0.95)

class TestMultiGPUPerformance(unittest.TestCase):
    """Test multi-GPU performance."""
    
    def setUp(self):
        """Create test data."""
        self.d = 128
        self.nb = 100000
        self.nq = 1000
        self.nlist = 1000
        self.k = 100
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        self.xq = make_data(self.nq, self.d)
        
    def test_sharding_speedup(self):
        """Test search speedup with sharding."""
        # Create and train index
        index_cpu = IVFFlatIndex(self.d, self.nlist)
        index_cpu.train(self.xb)
        index_cpu.add(self.xb)
        index_cpu.nprobe = 16
        
        # Time single-GPU search
        resources = [Resources()]
        index_gpu1 = index_cpu.to_gpus(resources)
        
        t0 = time.time()
        d1, i1 = index_gpu1.search(self.xq, self.k)
        t1 = time.time()
        single_gpu_time = t1 - t0
        
        # Time sharded search
        resources = [Resources() for _ in range(2)]
        index_gpu2 = index_cpu.to_gpus(
            resources,
            shard=True,
            share_quantizer=True
        )
        
        t0 = time.time()
        d2, i2 = index_gpu2.search(self.xq, self.k)
        t1 = time.time()
        multi_gpu_time = t1 - t0
        
        # Should see speedup with sharding
        self.assertLess(multi_gpu_time, single_gpu_time * 0.7)
        
        # Results should still be accurate
        match_ratio = (i1 == i2).sum() / i1.size
        self.assertGreater(match_ratio, 0.95)

if __name__ == '__main__':
    unittest.main()
