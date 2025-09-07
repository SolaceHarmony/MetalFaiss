"""
test_mlx_serialize.py - Tests for MLX index serialization

These tests verify that indices can be properly serialized and deserialized
while preserving their state and search results.
"""

import unittest
from .mlx_test_utils import assert_array_equal, assert_allclose
import mlx.core as mx
from typing import List, Tuple
import pickle
import tempfile
import os

from ..faissmlx.ops import array
from ..faissmlx.gpu_ops import GpuResources
from ..index.flat_index import FlatIndex
from ..index.ivf_flat_index import IVFFlatIndex
from ..index.product_quantizer_index import ProductQuantizerIndex
from ..index.scalar_quantizer_index import ScalarQuantizerIndex
from ..index.binary_flat_index import BinaryFlatIndex
from ..index.binary_ivf_index import BinaryIVFIndex

def make_data(num: int, d: int) -> mx.array:
    """Generate test data."""
    return array(mx.random.uniform(shape=(num, d)).astype(mx.float32).tolist())

def make_binary_data(num: int, d: int) -> mx.array:
    """Generate binary test data."""
    return array(mx.random.randint(0, 2, shape=(num, d), dtype=mx.uint8).tolist())

class TestMLXSerialize(unittest.TestCase):
    """Test MLX index serialization."""
    
    def setUp(self):
        """Create test data."""
        self.d = 32
        self.k = 10
        self.nlist = 5
        
        # Create test vectors
        self.train = make_data(10000, self.d)
        self.add = make_data(10000, self.d)
        self.query = make_data(10, self.d)
        
        # Create binary vectors
        self.d_bin = 64
        self.train_bin = make_binary_data(10000, self.d_bin)
        self.add_bin = make_binary_data(10000, self.d_bin)
        self.query_bin = make_binary_data(10, self.d_bin)
        
    def _test_index_serialization(self, index, is_binary: bool = False):
        """Test serialization for a specific index."""
        # Get training/test data
        train = self.train_bin if is_binary else self.train
        add = self.add_bin if is_binary else self.add
        query = self.query_bin if is_binary else self.query
        
        # Train and add
        index.train(train)
        index.add(add)
        
        # Get original search results
        d_orig, i_orig = index.search(query, self.k)
        
        # Serialize to file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                # Save
                pickle.dump(index, f)
                f.close()
                
                # Load
                with open(f.name, 'rb') as f2:
                    index_restore = pickle.load(f2)
                    
                # Search with restored index
                d_restore, i_restore = index_restore.search(query, self.k)
                
                # Results should match
                assert_array_equal(i_orig, i_restore)
                assert_allclose(d_orig, d_restore)
                
                # Should be able to add more vectors
                index_restore.add(query)
                
            finally:
                os.unlink(f.name)
                
    def test_flat_serialize(self):
        """Test flat index serialization."""
        # CPU index
        index = FlatIndex(self.d)
        self._test_index_serialization(index)
        
        # GPU index
        resources = GpuResources()
        index = FlatIndex(self.d).to_gpu(resources)
        self._test_index_serialization(index)
        
    def test_ivf_serialize(self):
        """Test IVF index serialization."""
        # CPU index
        index = IVFFlatIndex(self.d, self.nlist)
        self._test_index_serialization(index)
        
        # GPU index
        resources = GpuResources()
        index = IVFFlatIndex(self.d, self.nlist).to_gpu(resources)
        self._test_index_serialization(index)
        
    def test_pq_serialize(self):
        """Test PQ index serialization."""
        # CPU index
        index = ProductQuantizerIndex(self.d, self.nlist, M=4)
        self._test_index_serialization(index)
        
        # GPU index
        resources = GpuResources()
        index = ProductQuantizerIndex(self.d, self.nlist, M=4).to_gpu(resources)
        self._test_index_serialization(index)
        
    def test_sq_serialize(self):
        """Test scalar quantizer index serialization."""
        # CPU index
        index = ScalarQuantizerIndex(self.d, self.nlist)
        self._test_index_serialization(index)
        
        # GPU index
        resources = GpuResources()
        index = ScalarQuantizerIndex(self.d, self.nlist).to_gpu(resources)
        self._test_index_serialization(index)
        
    def test_binary_flat_serialize(self):
        """Test binary flat index serialization."""
        # CPU index
        index = BinaryFlatIndex(self.d_bin)
        self._test_index_serialization(index, is_binary=True)
        
        # GPU index
        resources = GpuResources()
        index = BinaryFlatIndex(self.d_bin).to_gpu(resources)
        self._test_index_serialization(index, is_binary=True)
        
    def test_binary_ivf_serialize(self):
        """Test binary IVF index serialization."""
        # CPU index
        index = BinaryIVFIndex(self.d_bin, self.nlist)
        self._test_index_serialization(index, is_binary=True)
        
        # GPU index
        resources = GpuResources()
        index = BinaryIVFIndex(self.d_bin, self.nlist).to_gpu(resources)
        self._test_index_serialization(index, is_binary=True)
        
    def test_large_index_serialize(self):
        """Test serialization with large indices."""
        # Create large dataset
        nb = 100000
        add_large = make_data(nb, self.d)
        
        # Test with flat index
        index = FlatIndex(self.d)
        index.add(add_large)
        self._test_index_serialization(index)
        
        # Test with IVF index
        index = IVFFlatIndex(self.d, self.nlist)
        index.train(self.train)
        index.add(add_large)
        self._test_index_serialization(index)
        
    def test_trained_only_serialize(self):
        """Test serialization of trained-only indices."""
        # Create index
        index = IVFFlatIndex(self.d, self.nlist)
        
        # Train but don't add
        index.train(self.train)
        
        # Serialize
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                # Save
                pickle.dump(index, f)
                f.close()
                
                # Load
                with open(f.name, 'rb') as f2:
                    index_restore = pickle.load(f2)
                    
                # Should be trained
                self.assertTrue(index_restore.is_trained)
                
                # Should be able to add vectors
                index_restore.add(self.add)
                
            finally:
                os.unlink(f.name)

if __name__ == '__main__':
    unittest.main()
