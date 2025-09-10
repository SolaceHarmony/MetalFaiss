"""
test_mlx_demo.py - Demonstration tests for MetalFaiss

These tests demonstrate common usage patterns and serve as examples for:
1. Basic index operations
2. GPU acceleration
3. Multi-GPU usage
4. Binary search
"""

import unittest
import mlx.core as mx
from typing import List, Tuple
import time

from ..faissmlx.ops import array
from ..faissmlx.resources import Resources
from ..index.flat_index import FlatIndex
from ..index.ivf_flat_index import IVFFlatIndex
from ..index.product_quantizer_index import ProductQuantizerIndex
from ..index.binary_flat_index import BinaryFlatIndex
from ..index.binary_ivf_index import BinaryIVFIndex

from ..utils.rng_utils import new_key

def make_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate test data using MLX RNG keys."""
    k = new_key(seed)
    return mx.random.uniform(shape=(num, d), key=k).astype(mx.float32)

def make_binary_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate binary test data."""
    k = new_key(seed)
    u = mx.random.uniform(shape=(num, d), key=k)
    return (u >= mx.array(0.5, dtype=u.dtype)).astype(mx.uint8)

class TestBasicUsage(unittest.TestCase):
    """Demonstrate basic index usage."""
    
    def test_flat_index(self):
        """Demo flat index usage."""
        # Create some test data
        d = 64          # dimensions
        nb = 100000    # database size
        nq = 10000     # queries
        k = 4          # number of nearest neighbors
        
        print("\nFlat Index Demo:")
        print(f"Dimensions: {d}")
        print(f"Database size: {nb}")
        print(f"Query size: {nq}")
        
        # Generate data
        xb = make_data(nb, d)
        xq = make_data(nq, d)
        
        # Create and add to index
        index = FlatIndex(d)
        t0 = time.time()
        index.add(xb)
        t1 = time.time()
        print(f"Add time: {t1-t0:.3f}s")
        
        # Search
        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()
        print(f"Search time: {t1-t0:.3f}s")
        print(f"First query results, distances: {D[0]}")
        print(f"First query results, indices: {I[0]}")

class TestGPUUsage(unittest.TestCase):
    """Demonstrate GPU acceleration."""
    
    def test_gpu_index(self):
        """Demo GPU-accelerated index."""
        # Create some test data
        d = 64          # dimensions
        nb = 100000    # database size
        nq = 10000     # queries
        k = 4          # number of nearest neighbors
        
        print("\nGPU Index Demo:")
        print(f"Dimensions: {d}")
        print(f"Database size: {nb}")
        print(f"Query size: {nq}")
        
        # Generate data
        xb = make_data(nb, d)
        xq = make_data(nq, d)
        
        # Create CPU index for comparison
        index_cpu = FlatIndex(d)
        t0 = time.time()
        index_cpu.add(xb)
        t1 = time.time()
        print(f"CPU add time: {t1-t0:.3f}s")
        
        t0 = time.time()
        D_cpu, I_cpu = index_cpu.search(xq, k)
        t1 = time.time()
        print(f"CPU search time: {t1-t0:.3f}s")
        
        # Create GPU index
        resources = Resources()
        index_gpu = index_cpu.to_gpu(resources)
        
        t0 = time.time()
        D_gpu, I_gpu = index_gpu.search(xq, k)
        t1 = time.time()
        print(f"GPU search time: {t1-t0:.3f}s")
        
        # Results should match
        self.assertTrue(bool(mx.all(mx.equal(I_cpu, I_gpu)).item()))

class TestMultiGPUUsage(unittest.TestCase):
    """Demonstrate multi-GPU usage."""
    
    def test_sharded_index(self):
        """Demo sharded index across GPUs."""
        # Create some test data
        d = 64          # dimensions
        nb = 1000000   # larger database
        nq = 10000     # queries
        k = 4          # number of nearest neighbors
        
        print("\nMulti-GPU Index Demo:")
        print(f"Dimensions: {d}")
        print(f"Database size: {nb}")
        print(f"Query size: {nq}")
        
        # Generate data
        xb = make_data(nb, d)
        xq = make_data(nq, d)
        
        # Create IVF index for better scaling
        nlist = 100
        index_cpu = IVFFlatIndex(d, nlist)
        
        # Train and add
        t0 = time.time()
        index_cpu.train(xb)
        index_cpu.add(xb)
        t1 = time.time()
        print(f"CPU add time: {t1-t0:.3f}s")
        
        # Search on CPU
        t0 = time.time()
        D_cpu, I_cpu = index_cpu.search(xq, k)
        t1 = time.time()
        print(f"CPU search time: {t1-t0:.3f}s")
        
        # Create sharded GPU index
        resources = [Resources() for _ in range(2)]
        index_gpu = index_cpu.to_gpus(
            resources,
            shard=True,
            share_quantizer=True
        )
        
        # Search on GPUs
        t0 = time.time()
        D_gpu, I_gpu = index_gpu.search(xq, k)
        t1 = time.time()
        print(f"Multi-GPU search time: {t1-t0:.3f}s")
        
        # Results should be very close
        match_ratio = float(mx.sum(mx.equal(I_gpu, I_cpu)).item()) / float(I_cpu.size)
        print(f"Result match ratio: {match_ratio:.3f}")

class TestBinaryUsage(unittest.TestCase):
    """Demonstrate binary index usage."""
    
    def test_binary_search(self):
        """Demo binary vector search."""
        # Create some test data
        d = 256        # dimensions (multiple of 8)
        nb = 100000    # database size
        nq = 10000     # queries
        k = 4          # number of nearest neighbors
        
        print("\nBinary Index Demo:")
        print(f"Dimensions: {d}")
        print(f"Database size: {nb}")
        print(f"Query size: {nq}")
        
        # Generate binary data
        xb = make_binary_data(nb, d)
        xq = make_binary_data(nq, d)
        
        # Create and add to CPU index
        index_cpu = BinaryFlatIndex(d)
        t0 = time.time()
        index_cpu.add(xb)
        t1 = time.time()
        print(f"CPU add time: {t1-t0:.3f}s")
        
        t0 = time.time()
        D_cpu, I_cpu = index_cpu.search(xq, k)
        t1 = time.time()
        print(f"CPU search time: {t1-t0:.3f}s")
        
        # Create GPU index
        resources = Resources()
        index_gpu = index_cpu.to_gpu(resources)
        
        t0 = time.time()
        D_gpu, I_gpu = index_gpu.search(xq, k)
        t1 = time.time()
        print(f"GPU search time: {t1-t0:.3f}s")
        
        # Results should match exactly
        self.assertTrue(bool(mx.all(mx.equal(I_cpu, I_gpu)).item()))
        self.assertTrue(bool(mx.all(mx.equal(D_cpu, D_gpu)).item()))

if __name__ == '__main__':
    unittest.main(verbosity=2)
