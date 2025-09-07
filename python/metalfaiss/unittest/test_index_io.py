"""
test_index_io.py - Tests for index I/O
"""

import os
import unittest
import tempfile
from .mlx_test_utils import assert_array_equal, assert_allclose
import mlx.core as mx
from typing import List, Optional, Tuple

from ..index.flat_index import FlatIndex
from ..index.ivf_flat_index import IVFFlatIndex
from ..index.binary_flat_index import BinaryFlatIndex
from ..index.binary_ivf_index import BinaryIVFIndex
from ..index.product_quantizer_index import ProductQuantizerIndex
from ..index.scalar_quantizer_index import ScalarQuantizerIndex
from ..index.index_io import write_index, read_index, IOFlag
from ..types.metric_type import MetricType
from ..types.quantizer_type import QuantizerType

class TestIndexIO(unittest.TestCase):
    """Test index I/O."""
    
    def setUp(self):
        """Set up test data."""
        self.d = 64
        self.n = 100
        self.x = mx.random.normal(shape=(self.n, self.d)).astype(mx.float32)
        self.xb = mx.random.randint(0, 2, shape=(self.n, self.d), dtype=mx.uint8)
        
    def test_flat_index(self):
        """Test flat index I/O."""
        index = FlatIndex(self.d)
        index.add(self.x)
        
        # Save and load
        with tempfile.NamedTemporaryFile() as f:
            write_index(index, f.name)
            index2 = read_index(f.name)
            
        # Check properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(len(index2), self.n)
        self.assertEqual(index2.metric, MetricType.L2)
        
        # Check search results match
        k = 4
        D1, I1 = index.search(self.x[:10], k)
        D2, I2 = index2.search(self.x[:10], k)
        assert_array_equal(D1, D2)
        assert_array_equal(I1, I2)
        
    def test_ivf_flat_index(self):
        """Test IVF flat index I/O."""
        nlist = 10
        index = IVFFlatIndex(self.d, nlist)
        index.train(self.x)
        index.add(self.x)
        
        # Save and load
        with tempfile.NamedTemporaryFile() as f:
            write_index(index, f.name)
            index2 = read_index(f.name)
            
        # Check properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(len(index2), self.n)
        self.assertEqual(index2.nlist, nlist)
        self.assertTrue(index2.is_trained)
        
        # Check search results match
        k = 4
        D1, I1 = index.search(self.x[:10], k)
        D2, I2 = index2.search(self.x[:10], k)
        assert_array_equal(D1, D2)
        assert_array_equal(I1, I2)
        
    def test_binary_flat_index(self):
        """Test binary flat index I/O."""
        index = BinaryFlatIndex(self.d)
        index.add(self.xb)
        
        # Save and load
        with tempfile.NamedTemporaryFile() as f:
            write_index(index, f.name)
            index2 = read_index(f.name)
            
        # Check properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(len(index2), self.n)
        
        # Check search results match
        k = 4
        D1, I1 = index.search(self.xb[:10], k)
        D2, I2 = index2.search(self.xb[:10], k)
        assert_array_equal(D1, D2)
        assert_array_equal(I1, I2)
        
    def test_binary_ivf_index(self):
        """Test binary IVF index I/O."""
        nlist = 10
        index = BinaryIVFIndex(self.d, nlist)
        index.train(self.xb)
        index.add(self.xb)
        
        # Save and load
        with tempfile.NamedTemporaryFile() as f:
            write_index(index, f.name)
            index2 = read_index(f.name)
            
        # Check properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(len(index2), self.n)
        self.assertEqual(index2.nlist, nlist)
        self.assertTrue(index2.is_trained)
        
        # Check search results match
        k = 4
        D1, I1 = index.search(self.xb[:10], k)
        D2, I2 = index2.search(self.xb[:10], k)
        np.testing.assert_array_equal(D1, D2)
        np.testing.assert_array_equal(I1, I2)
        
    def test_product_quantizer_index(self):
        """Test product quantizer index I/O."""
        M = 8
        index = ProductQuantizerIndex(self.d, M)
        index.train(self.x)
        index.add(self.x)
        
        # Save and load
        with tempfile.NamedTemporaryFile() as f:
            write_index(index, f.name)
            index2 = read_index(f.name)
            
        # Check properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(len(index2), self.n)
        self.assertEqual(index2.M, M)
        self.assertTrue(index2.is_trained)
        
        # Check search results match
        k = 4
        D1, I1 = index.search(self.x[:10], k)
        D2, I2 = index2.search(self.x[:10], k)
        assert_allclose(D1, D2)
        assert_array_equal(I1, I2)
        
    def test_scalar_quantizer_index(self):
        """Test scalar quantizer index I/O."""
        qtype = QuantizerType.QT_8bit
        index = ScalarQuantizerIndex(self.d, qtype)
        index.train(self.x)
        index.add(self.x)
        
        # Save and load
        with tempfile.NamedTemporaryFile() as f:
            write_index(index, f.name)
            index2 = read_index(f.name)
            
        # Check properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(len(index2), self.n)
        self.assertEqual(index2.qtype, qtype)
        self.assertTrue(index2.is_trained)
        
        # Check search results match
        k = 4
        D1, I1 = index.search(self.x[:10], k)
        D2, I2 = index2.search(self.x[:10], k)
        assert_allclose(D1, D2)
        assert_array_equal(I1, I2)

if __name__ == '__main__':
    unittest.main()
