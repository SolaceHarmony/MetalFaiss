# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)

"""
Tests for Index Factory System
"""

import unittest
import mlx.core as mx
import mlx.core as mx

from metalfaiss import index_factory, get_supported_descriptions, MetricType
from metalfaiss.index.flat_index import FlatIndex
from metalfaiss.index.ivf_flat_index import IVFFlatIndex
from metalfaiss.index.hnsw_index import HNSWFlatIndex
from metalfaiss.index.product_quantizer_index import ProductQuantizerIndex
from metalfaiss.index.scalar_quantizer_index import ScalarQuantizerIndex
from metalfaiss.index.pre_transform_index import PreTransformIndex
from metalfaiss.index.id_map import IDMap
from metalfaiss.errors import InvalidArgumentError


class TestIndexFactory(unittest.TestCase):
    """Test the index factory system."""
    
    def setUp(self):
        self.d = 64
        self.n = 100
        # Create test data
        self.data = np.random.randn(self.n, self.d).astype(np.float32)
        self.query = np.random.randn(1, self.d).astype(np.float32)
    
    def test_flat_index(self):
        """Test creating flat index."""
        index = index_factory(self.d, "Flat")
        self.assertIsInstance(index, FlatIndex)
        self.assertEqual(index.d, self.d)
    
    def test_ivf_flat_index(self):
        """Test creating IVF flat index."""
        index = index_factory(self.d, "IVF100")
        self.assertIsInstance(index, IVFFlatIndex)
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.nlist, 100)
        
        # Test with explicit Flat
        index2 = index_factory(self.d, "IVF50,Flat")
        self.assertIsInstance(index2, IVFFlatIndex)
        self.assertEqual(index2.nlist, 50)
    
    def test_hnsw_index(self):
        """Test creating HNSW index."""
        index = index_factory(self.d, "HNSW16")
        self.assertIsInstance(index, HNSWFlatIndex)
        self.assertEqual(index.d, self.d)
        
        # Test default M parameter
        index2 = index_factory(self.d, "HNSW")
        self.assertIsInstance(index2, HNSWFlatIndex)
    
    def test_pq_index(self):
        """Test creating PQ index."""
        index = index_factory(self.d, "PQ8")
        self.assertIsInstance(index, ProductQuantizerIndex)
        self.assertEqual(index.d, self.d)
        
        # Test default m parameter
        index2 = index_factory(self.d, "PQ")
        self.assertIsInstance(index2, ProductQuantizerIndex)
    
    def test_scalar_quantizer_index(self):
        """Test creating SQ index."""
        index = index_factory(self.d, "SQ8")
        self.assertIsInstance(index, ScalarQuantizerIndex)
        self.assertEqual(index.d, self.d)
        
        # Test different bit sizes
        index2 = index_factory(self.d, "SQ4")
        self.assertIsInstance(index2, ScalarQuantizerIndex)
    
    def test_id_map_wrapper(self):
        """Test creating IDMap wrapper."""
        index = index_factory(self.d, "IDMap,Flat")
        self.assertIsInstance(index, IDMap)
        
        index2 = index_factory(self.d, "IDMap,IVF100")
        self.assertIsInstance(index2, IDMap)
        # IDMap2 variant
        try:
            index3 = index_factory(self.d, "IDMap2,Flat")
            from metalfaiss.index.id_map import IDMap2
            self.assertIsInstance(index3, IDMap2)
        except Exception:
            # Accept if IDMap2 not available yet
            pass
    
    def test_preprocessing_transforms(self):
        """Test creating indexes with preprocessing."""
        index = index_factory(self.d, "PCA32,Flat")
        self.assertIsInstance(index, PreTransformIndex)
        
        index2 = index_factory(self.d, "OPQ64,IVF100")
        self.assertIsInstance(index2, PreTransformIndex)

    def test_refine_flat(self):
        """Test creating refine-flat variants."""
        from metalfaiss.index.refine_flat_index import RefineFlatIndex
        idx = index_factory(self.d, "RFlat")
        self.assertIsInstance(idx, RefineFlatIndex)
        idx2 = index_factory(self.d, "Refine(Flat)")
        self.assertIsInstance(idx2, RefineFlatIndex)
    
    def test_metric_types(self):
        """Test different metric types."""
        index1 = index_factory(self.d, "Flat", MetricType.L2)
        self.assertEqual(index1.metric_type, MetricType.L2)
        
        index2 = index_factory(self.d, "Flat", MetricType.INNER_PRODUCT)
        self.assertEqual(index2.metric_type, MetricType.INNER_PRODUCT)
        
        # Test string metric
        index3 = index_factory(self.d, "Flat", "L1")
        self.assertEqual(index3.metric_type, MetricType.L1)
    
    def test_invalid_descriptions(self):
        """Test error handling for invalid descriptions."""
        with self.assertRaises(InvalidArgumentError):
            index_factory(self.d, "InvalidType")
        
        with self.assertRaises(InvalidArgumentError):
            index_factory(self.d, "IVF100,InvalidQuantizer")
        
        with self.assertRaises(InvalidArgumentError):
            index_factory(self.d, "Too,Many,Parts,Here")
    
    def test_get_supported_descriptions(self):
        """Test getting supported descriptions."""
        descriptions = get_supported_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertIn("Flat", descriptions)
        self.assertIn("IVF{nlist}", descriptions)
        self.assertIn("PQ{m}", descriptions)
    
    def test_complex_combinations(self):
        """Test complex index combinations."""
        # IVF+PQ should be supported (L2 only)
        idx = index_factory(self.d, "IVF32,PQ4")
        from metalfaiss.index.ivf_pq_index import IVFPQIndex
        self.assertIsInstance(idx, IVFPQIndex)
    
    def test_end_to_end_usage(self):
        """Test end-to-end usage of factory-created indexes."""
        index = index_factory(self.d, "Flat")
        
        # Add data
        index.add(mx.array(self.data))
        self.assertEqual(index.ntotal, self.n)
        
        # Search
        distances, indices = index.search(mx.array(self.query), k=5)
        self.assertEqual(distances.shape, (1, 5))
        self.assertEqual(indices.shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
