"""
test_mlx_gpu_edge.py - Tests for GPU edge cases

These tests verify behavior in edge cases like:
1. Memory exhaustion
2. Code packing/unpacking
3. Error handling
4. Resource management
"""

import unittest
import mlx.core as mx
from typing import List, Tuple

from ..faissmlx.ops import array
from ..faissmlx.resources import (
    Resources,
    MemoryManager,
)
from ..index.flat_index import FlatIndex
from ..index.ivf_flat_index import IVFFlatIndex
from ..index.product_quantizer_index import ProductQuantizerIndex

def make_data(num: int, d: int, seed: int = 42) -> mx.array:
    """Generate test data (MLX)."""
    k = mx.random.key(int(seed))
    return array(mx.random.uniform(shape=(num, d), key=k).astype(mx.float32))

class TestGpuMemory(unittest.TestCase):
    """Test GPU memory management."""
    
    def setUp(self):
        """Create test data."""
        self.d = 256
        self.resources = Resources(max_memory=1024 * 1024)  # 1MB limit
        self.manager = MemoryManager(self.resources)
        
    def test_memory_limit(self):
        """Test memory limit enforcement."""
        with self.manager:
            # Should succeed - small allocation
            arr = self.manager.alloc((1000,), dtype="float32")
            self.assertEqual(arr.shape, (1000,))
            
            # Should fail - too large
            with self.assertRaises(MemoryError):
                self.manager.alloc((1000000,), dtype="float32")
                
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        with self.manager:
            # Allocate array
            arr1 = self.manager.alloc((1000,), dtype="float32")
            initial_mem = self.resources.current_memory
            
            # Allocate another
            arr2 = self.manager.alloc((1000,), dtype="float32")
            self.assertGreater(
                self.resources.current_memory,
                initial_mem
            )
            
            # Free first array
            self.manager.free(arr1)
            self.assertLess(
                self.resources.current_memory,
                initial_mem * 2
            )
            
    def test_memory_cleanup(self):
        """Test memory cleanup on context exit."""
        with self.manager:
            arr = self.manager.alloc((1000,), dtype="float32")
            
        # Should be reset after context
        self.assertEqual(self.resources.current_memory, 0)
        
    def test_out_of_memory_recovery(self):
        """Test recovery from out of memory."""
        with self.manager:
            # Trigger OOM
            with self.assertRaises(MemoryError):
                self.manager.alloc((1000000,), dtype="float32")
                
            # Should still be able to allocate small arrays
            arr = self.manager.alloc((100,), dtype="float32")
            self.assertEqual(arr.shape, (100,))

class TestCodePacking(unittest.TestCase):
    """Test code packing/unpacking."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64
        self.nb = 1000
        self.nlist = 100
        self.M = 8  # Number of sub-quantizers
        self.nbits = 8  # Bits per sub-quantizer
        
        # Create test vectors
        self.xb = make_data(self.nb, self.d)
        
    def test_pq_packing(self):
        """Test PQ code packing."""
        # Create CPU index
        index = ProductQuantizerIndex(
            self.d,
            self.nlist,
            M=self.M,
            nbits=self.nbits
        )
        
        # Train and add
        index.train(self.xb)
        index.add(self.xb)
        
        # Get codes
        codes = index.pq.compute_codes(self.xb)
        
        # Pack codes
        packed = index.pq.pack_codes(codes)
        
        # Unpack codes
        unpacked = index.pq.unpack_codes(packed)
        
        # Should match original
        from .mlx_test_utils import assert_array_equal
        assert_array_equal(codes, unpacked)
        
    def test_pq_partial_packing(self):
        """Test partial PQ code packing."""
        # Create CPU index
        index = ProductQuantizerIndex(
            self.d,
            self.nlist,
            M=self.M,
            nbits=self.nbits
        )
        
        # Train and add
        index.train(self.xb)
        index.add(self.xb)
        
        # Get codes for subset
        subset = self.xb[:100]
        codes = index.pq.compute_codes(subset)
        
        # Pack subset
        packed = index.pq.pack_codes(codes)
        
        # Unpack subset
        unpacked = index.pq.unpack_codes(packed)
        
        # Should match original subset
        from .mlx_test_utils import assert_array_equal
        assert_array_equal(codes, unpacked)

class TestResourceManagement(unittest.TestCase):
    """Test GPU resource management."""
    
    def setUp(self):
        """Create resources."""
        self.resources = [
            Resources(max_memory=1024 * 1024)
            for _ in range(2)
        ]
        
    def test_resource_isolation(self):
        """Test resource isolation."""
        # Allocate on first GPU
        with MemoryManager(self.resources[0]):
            arr1 = self.resources[0].alloc((1000,), dtype="float32")
            mem1 = self.resources[0].current_memory
            
            # Second GPU should be unaffected
            self.assertEqual(self.resources[1].current_memory, 0)
            
        # First GPU should be cleaned up
        self.assertEqual(self.resources[0].current_memory, 0)
        
    def test_resource_limits(self):
        """Test per-resource memory limits."""
        # Set different limits
        self.resources[0].max_memory = 1024 * 1024  # 1MB
        self.resources[1].max_memory = 2 * 1024 * 1024  # 2MB
        
        # First GPU should fail sooner
        with MemoryManager(self.resources[0]):
            with self.assertRaises(MemoryError):
                arr = self.resources[0].alloc((1100000,), dtype="float32")
                
        # Second GPU should succeed
        with MemoryManager(self.resources[1]):
            arr = self.resources[1].alloc((1100000,), dtype="float32")
            self.assertEqual(arr.shape, (1100000,))

class TestErrorHandling(unittest.TestCase):
    """Test GPU error handling."""
    
    def setUp(self):
        """Create resources."""
        self.resources = Resources()
        
    def test_invalid_device(self):
        """Test invalid device handling."""
        with self.assertRaises(ValueError):
            Resources(device_id=-1)
            
        with self.assertRaises(ValueError):
            Resources(device_id=1000)
            
    def test_invalid_memory(self):
        """Test invalid memory configuration."""
        with self.assertRaises(ValueError):
            Resources(max_memory=-1)
            
    # GPU-only design: transfer/resource exhaustion tests removed.

if __name__ == '__main__':
    unittest.main()
