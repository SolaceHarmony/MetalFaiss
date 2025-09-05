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
import numpy as np
from ..utils.search_result import SearchResult, SearchRangeResult

class TestSearchResult(unittest.TestCase):
    """Test k-NN search result functionality."""
    
    def setUp(self):
        """Create test data."""
        self.nq = 3  # Number of queries
        self.k = 4   # Number of neighbors
        
        # Create sample results
        self.distances = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6]
        ]
        self.labels = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5]
        ]
        self.result = SearchResult(self.distances, self.labels)
        
    def test_properties(self):
        """Test basic properties."""
        self.assertEqual(self.result.nq, self.nq)
        self.assertEqual(self.result.k, self.k)
        
    def test_indexing(self):
        """Test result indexing."""
        # Get single query results
        distances, labels = self.result[0]
        self.assertEqual(distances, self.distances[0])
        self.assertEqual(labels, self.labels[0])
        
        # Check all queries
        for i in range(self.nq):
            distances, labels = self.result[i]
            self.assertEqual(distances, self.distances[i])
            self.assertEqual(labels, self.labels[i])
            
    def test_array_conversion(self):
        """Test conversion to/from MLX arrays."""
        # Convert to arrays
        distances, labels = self.result.to_arrays()
        
        # Check shapes
        self.assertEqual(distances.shape, (self.nq, self.k))
        self.assertEqual(labels.shape, (self.nq, self.k))
        
        # Check values
        np.testing.assert_array_almost_equal(
            distances.numpy(),
            np.array(self.distances)
        )
        np.testing.assert_array_equal(
            labels.numpy(),
            np.array(self.labels)
        )
        
        # Convert back
        result2 = SearchResult.from_arrays(distances, labels)
        self.assertEqual(result2.distances, self.distances)
        self.assertEqual(result2.labels, self.labels)

class TestSearchRangeResult(unittest.TestCase):
    """Test range search result functionality."""
    
    def setUp(self):
        """Create test data."""
        self.nq = 3  # Number of queries
        
        # Create sample results with variable numbers of neighbors
        self.lims = [0, 2, 5, 7]  # Query 0: 2 neighbors, Query 1: 3 neighbors, Query 2: 2 neighbors
        self.distances = [
            [0.1, 0.2],           # Query 0
            [0.2, 0.3, 0.4],      # Query 1
            [0.3, 0.4]            # Query 2
        ]
        self.labels = [
            [0, 1],               # Query 0
            [1, 2, 3],           # Query 1
            [2, 3]               # Query 2
        ]
        self.result = SearchRangeResult(
            self.lims,
            self.distances,
            self.labels
        )
        
    def test_properties(self):
        """Test basic properties."""
        self.assertEqual(self.result.nq, self.nq)
        
    def test_indexing(self):
        """Test result indexing."""
        # Check each query
        for i in range(self.nq):
            distances, labels = self.result[i]
            self.assertEqual(distances, self.distances[i])
            self.assertEqual(labels, self.labels[i])
            
    def test_array_conversion(self):
        """Test conversion to/from MLX arrays."""
        # Convert to arrays
        lims, distances, labels = self.result.to_arrays()
        
        # Check shapes
        self.assertEqual(lims.shape, (self.nq + 1,))
        self.assertEqual(distances.shape, (self.lims[-1],))
        self.assertEqual(labels.shape, (self.lims[-1],))
        
        # Check values
        np.testing.assert_array_equal(
            lims.numpy(),
            np.array(self.lims)
        )
        np.testing.assert_array_almost_equal(
            distances.numpy(),
            np.array([d for dists in self.distances for d in dists])
        )
        np.testing.assert_array_equal(
            labels.numpy(),
            np.array([l for labs in self.labels for l in labs])
        )
        
        # Convert back
        result2 = SearchRangeResult.from_arrays(lims, distances, labels)
        self.assertEqual(result2.lims, self.lims)
        self.assertEqual(result2.distances, self.distances)
        self.assertEqual(result2.labels, self.labels)
        
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
        self.assertEqual(merged.lims, [0, 3, 8, 11])  # 3, 5, 3 neighbors
        
        # Check sorting by distance
        for i in range(self.nq):
            distances, _ = merged[i]
            self.assertEqual(distances, sorted(distances))
            
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