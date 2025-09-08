"""
test_search_result.py - Tests for search result functionality

These tests verify that our search result implementation matches FAISS,
particularly around:
- K-NN search results
- Range search results
- Array conversions
- Result merging
"""

import unittest
import mlx.core as mx
from ..utils.search_result import SearchResult, SearchRangeResult

class TestSearchResult(unittest.TestCase):
    """Test k-NN search result functionality."""
    
    def setUp(self):
        """Create test data."""
        self.nq = 3  # Number of queries
        self.k = 4   # Number of neighbors
        
        # Create sample results
        self.distances = mx.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6]
        ], dtype=mx.float32)
        self.labels = mx.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5]
        ], dtype=mx.int32)
        self.result = SearchResult(self.distances, self.labels)
        
    def test_properties(self):
        """Test basic properties."""
        self.assertEqual(self.result.nq, self.nq)
        self.assertEqual(self.result.k, self.k)
        
    def test_indexing(self):
        """Test result indexing."""
        # Get single query results
        r0 = self.result[0]
        self.assertTrue(bool(mx.allclose(r0.distances, self.distances[0], rtol=1e-6, atol=1e-6).item()))  # boundary-ok
        self.assertTrue(bool(mx.all(mx.equal(r0.indices, self.labels[0])).item()))  # boundary-ok
        
        # Check all queries
        for i in range(self.nq):
            r = self.result[i]
            self.assertTrue(bool(mx.allclose(r.distances, self.distances[i], rtol=1e-6, atol=1e-6).item()))  # boundary-ok
            self.assertTrue(bool(mx.all(mx.equal(r.indices, self.labels[i])).item()))  # boundary-ok
            
    def test_array_conversion(self):
        """Test conversion to/from MLX arrays."""
        # Shapes and values already checked above
        self.assertEqual(self.result.distances.shape, (self.nq, self.k))
        self.assertEqual(self.result.indices.shape, (self.nq, self.k))
        self.assertTrue(bool(mx.allclose(self.result.distances, self.distances, rtol=1e-6, atol=1e-6).item()))  # boundary-ok
        self.assertTrue(bool(mx.all(mx.equal(self.result.indices, self.labels)).item()))  # boundary-ok

class TestSearchRangeResult(unittest.TestCase):
    """Test range search result functionality."""
    
    def setUp(self):
        """Create test data."""
        self.nq = 3  # Number of queries
        
        # Create sample results with variable numbers of neighbors
        self.lims = mx.array([0, 2, 5, 7], dtype=mx.int32)
        self.distances = [
            mx.array([0.1, 0.2], dtype=mx.float32),
            mx.array([0.2, 0.3, 0.4], dtype=mx.float32),
            mx.array([0.3, 0.4], dtype=mx.float32)
        ]
        self.labels = [
            mx.array([0, 1], dtype=mx.int32),
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([2, 3], dtype=mx.int32)
        ]
        self.result = SearchRangeResult(
            distances=self.distances,
            indices=self.labels,
            lims=self.lims
        )
        
    def test_properties(self):
        """Test basic properties."""
        self.assertEqual(self.result.nq, self.nq)
        
    def test_indexing(self):
        """Test result indexing."""
        # Check each query
        for i in range(self.nq):
            distances, labels = self.result[i]
            self.assertTrue(bool(mx.allclose(distances, self.distances[i], rtol=1e-6, atol=1e-6).item()))  # boundary-ok
            self.assertTrue(bool(mx.all(mx.equal(labels, self.labels[i])).item()))  # boundary-ok
            
    def test_array_conversion(self):
        """Test conversion to/from MLX arrays."""
        # Convert to arrays
        # Basic checks
        self.assertEqual(self.result.lims.shape, (self.nq + 1,))
        self.assertEqual(int(self.result.lims[-1].item()), sum(len(d) for d in self.distances))  # boundary-ok
        
    def test_merging(self):
        """Test result merging."""
        # Create another result with same queries
        other = SearchRangeResult(
            lims=[0, 1, 3, 4],
            distances=[
                [0.15],           # Query 0
                [0.25, 0.35],     # Query 1
                [0.45]            # Query 2
            ],
            labels=[
                [5],              # Query 0
                [6, 7],          # Query 1
                [8]              # Query 2
            ]
        )
        
        # Merge results
        merged = self.result.merge(other)
        
        # Check number of neighbors
        self.assertTrue(bool(mx.all(mx.equal(merged.lims, mx.array([0,3,8,11], dtype=mx.int32))).item()))  # boundary-ok
        
        # Check sorting by distance
        for i in range(self.nq):
            distances, _ = merged[i]
            self.assertTrue(bool(mx.all(mx.less_equal(distances[:-1], distances[1:])).item()))  # boundary-ok
            
        # Try merging with incompatible result
        other_bad = SearchRangeResult(
            lims=[0, 1],  # Only 1 query
            distances=[[0.1]],
            labels=[[0]]
        )
        with self.assertRaises(ValueError):
            self.result.merge(other_bad)
            
    def test_empty_results(self):
        """Test handling of empty results."""
        # Create result with no neighbors for some queries
        result = SearchRangeResult(
            lims=[0, 0, 2, 2],  # No neighbors for queries 0 and 2
            distances=[
                [],             # Query 0
                [0.1, 0.2],    # Query 1
                []             # Query 2
            ],
            labels=[
                [],            # Query 0
                [0, 1],       # Query 1
                []            # Query 2
            ]
        )
        
        # Check properties
        self.assertEqual(result.nq, 3)
        
        # Check empty queries
        distances, labels = result[0]
        self.assertEqual(distances, [])
        self.assertEqual(labels, [])
        
        # Convert to arrays
        lims, distances, labels = result.to_arrays()
        self.assertEqual(lims.shape, (4,))
        self.assertEqual(distances.shape, (2,))
        self.assertEqual(labels.shape, (2,))

if __name__ == '__main__':
    unittest.main()
