"""
test_id_map.py - Tests for ID mapping functionality

These tests verify that our ID mapping implementation matches FAISS,
particularly around:
- Basic ID translation
- Two-way mapping
- Vector removal
- Consistency checks
"""

import unittest
import mlx.core as mx
from typing import List
from ..index.id_map import IDMap, IDMap2, IDSelectorTranslated
from ..index.flat_index import FlatIndex
from ..types.metric_type import MetricType
from ..index.id_selector import IDSelectorBatch

class TestIDMap(unittest.TestCase):
    """Test basic ID mapping functionality."""
    
    def setUp(self):
        """Create test data."""
        # MLX-only random vectors
        self.d = 8  # Dimension
        self.n = 100  # Number of vectors
        
        # Create random vectors and IDs
        self.vectors = mx.random.normal(shape=(self.n, self.d)).astype(mx.float32)
        self.ids = list(range(1000, 1000 + self.n))  # External IDs
        
        # Create base index
        self.base_index = FlatIndex(self.d)
        
        # Create ID map
        self.index = IDMap(self.base_index)
        
    def test_add_with_ids(self):
        """Test adding vectors with IDs."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Check counts
        self.assertEqual(self.index.ntotal, self.n)
        self.assertEqual(len(self.index.id_map), self.n)
        
        # Check ID mapping
        self.assertEqual(self.index.id_map, self.ids)
        
    def test_add_without_ids(self):
        """Test that adding without IDs fails."""
        with self.assertRaises(Exception):
            self.index.add(self.vectors.tolist())
            
    def test_search(self):
        """Test search with ID translation."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Search
        k = 5
        queries = self.vectors[:3].tolist()  # Use first 3 vectors as queries
        result = self.index.search(queries, k)
        
        # Check result format
        self.assertEqual(len(result.distances), 3)
        self.assertEqual(len(result.labels), 3)
        self.assertEqual(len(result.distances[0]), k)
        self.assertEqual(len(result.labels[0]), k)
        
        # Check ID translation
        for labels in result.labels:
            for label in labels:
                self.assertTrue(label in self.ids or label == -1)
                
    def test_range_search(self):
        """Test range search with ID translation."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Range search
        radius = 1.0
        queries = self.vectors[:3].tolist()
        result = self.index.range_search(queries, radius)
        
        # Check result format
        self.assertEqual(len(result.lims), 4)  # nq + 1
        
        # Check ID translation
        for labels in result.labels:
            for label in labels:
                self.assertTrue(label in self.ids or label == -1)
                
    def test_remove_ids(self):
        """Test removing vectors by ID."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Create selector for even IDs
        to_remove = [id for id in self.ids if id % 2 == 0]
        selector = IDSelectorBatch(to_remove)
        
        # Remove vectors
        n_removed = self.index.remove_ids(selector)
        
        # Check removal
        self.assertEqual(n_removed, len(to_remove))
        self.assertEqual(self.index.ntotal, self.n - n_removed)
        
        # Check remaining IDs
        for id in self.index.id_map:
            self.assertTrue(id % 2 == 1)  # Only odd IDs remain
            
    def test_reset(self):
        """Test index reset."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Reset
        self.index.reset()
        
        # Check state
        self.assertEqual(self.index.ntotal, 0)
        self.assertEqual(len(self.index.id_map), 0)

class TestIDMap2(unittest.TestCase):
    """Test two-way ID mapping functionality."""
    
    def setUp(self):
        """Create test data."""
        # MLX-only random vectors
        self.d = 8
        self.n = 100
        
        self.vectors = mx.random.normal(shape=(self.n, self.d)).astype(mx.float32)
        self.ids = list(range(1000, 1000 + self.n))
        
        self.base_index = FlatIndex(self.d)
        self.index = IDMap2(self.base_index)
        
    def test_add_with_ids(self):
        """Test adding vectors with two-way mapping."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Check mappings
        self.assertEqual(len(self.index.id_map), self.n)
        self.assertEqual(len(self.index.rev_map), self.n)
        
        # Check reverse mapping
        for i, id in enumerate(self.ids):
            self.assertEqual(self.index.rev_map[id], i)
            
    def test_duplicate_ids(self):
        """Test that duplicate IDs are rejected."""
        ids = self.ids.copy()
        ids[0] = ids[1]  # Create duplicate
        
        with self.assertRaises(ValueError):
            self.index.add_with_ids(self.vectors.tolist(), ids)
            
    def test_reconstruction(self):
        """Test vector reconstruction by ID."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Reconstruct each vector
        for i, id in enumerate(self.ids):
            reconstructed = self.index.reconstruct(id)
            self.assertEqual([round(v, 6) for v in reconstructed], [round(v, 6) for v in self.vectors[i].tolist()])
            
        # Try invalid ID
        with self.assertRaises(KeyError):
            self.index.reconstruct(9999)
            
    def test_remove_ids(self):
        """Test removing vectors with two-way mapping update."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Remove some vectors
        to_remove = self.ids[::2]  # Every other ID
        selector = IDSelectorBatch(to_remove)
        n_removed = self.index.remove_ids(selector)
        
        # Check removal
        self.assertEqual(n_removed, len(to_remove))
        self.assertEqual(self.index.ntotal, self.n - n_removed)
        
        # Check mappings are consistent
        self.index.check_consistency()
        
        # Check reconstruction still works
        for id in self.index.id_map:
            reconstructed = self.index.reconstruct(id)
            original_idx = self.ids.index(id)
            self.assertEqual([round(v, 6) for v in reconstructed], [round(v, 6) for v in self.vectors[original_idx].tolist()])
            
    def test_consistency_check(self):
        """Test consistency checking."""
        # Add vectors
        self.index.add_with_ids(self.vectors.tolist(), self.ids)
        
        # Check initial consistency
        self.index.check_consistency()
        
        # Break consistency
        self.index.rev_map[self.ids[0]] = 999
        with self.assertRaises(RuntimeError):
            self.index.check_consistency()

class TestIDSelectorTranslated(unittest.TestCase):
    """Test ID selector translation."""
    
    def test_selector_translation(self):
        """Test translating between internal and external IDs."""
        # Create ID mapping
        internal_ids = list(range(10))
        external_ids = list(range(1000, 1010))
        
        # Create selector for even external IDs
        to_select = [id for id in external_ids if id % 2 == 0]
        base_selector = IDSelectorBatch(to_select)
        
        # Create translated selector
        translated = IDSelectorTranslated(external_ids, base_selector)
        
        # Check selection
        for i in internal_ids:
            external_id = external_ids[i]
            self.assertEqual(
                translated.is_member(i),
                external_id % 2 == 0
            )

if __name__ == '__main__':
    unittest.main()
