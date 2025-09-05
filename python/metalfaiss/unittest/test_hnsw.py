"""
test_hnsw.py - Tests for HNSW implementation

These tests verify that our HNSW implementation matches the behavior of the
original FAISS implementation, particularly around:
- Level generation
- Neighbor selection
- Distance computation
- Search accuracy
"""

import unittest
import mlx.core as mx
import numpy as np
from typing import List, Tuple
from ..types.metric_type import MetricType
from ..index.hnsw import HNSW, MinimaxHeap, HNSWStats
from ..index.hnsw_index import HNSWIndex, HNSWFlatIndex

class TestMinimaxHeap(unittest.TestCase):
    """Test MinimaxHeap implementation."""
    
    def setUp(self):
        self.heap = MinimaxHeap(5)
        
    def test_push_pop(self):
        """Test basic push and pop operations."""
        # Push elements
        self.heap.push(1, 3.0)
        self.heap.push(2, 1.0)
        self.heap.push(3, 4.0)
        self.heap.push(4, 2.0)
        
        # Pop elements - should come out in ascending order
        id1, dist1 = self.heap.pop_min()
        id2, dist2 = self.heap.pop_min()
        id3, dist3 = self.heap.pop_min()
        id4, dist4 = self.heap.pop_min()
        
        self.assertEqual(id1, 2)  # id with distance 1.0
        self.assertEqual(id2, 4)  # id with distance 2.0
        self.assertEqual(id3, 1)  # id with distance 3.0
        self.assertEqual(id4, 3)  # id with distance 4.0
        
    def test_capacity(self):
        """Test heap respects capacity limit."""
        # Push more elements than capacity
        for i in range(10):
            self.heap.push(i, float(i))
            
        self.assertEqual(self.heap.n, 5)  # Should only keep 5 elements
        
        # Elements should be the 5 smallest
        distances = []
        while self.heap.nvalid > 0:
            _, dist = self.heap.pop_min()
            distances.append(dist)
            
        self.assertEqual(distances, [0.0, 1.0, 2.0, 3.0, 4.0])
        
    def test_invalid_elements(self):
        """Test handling of invalid (-1) elements."""
        self.heap.push(1, 1.0)
        self.heap.push(-1, 2.0)  # Should be ignored
        self.heap.push(3, 3.0)
        
        self.assertEqual(self.heap.nvalid, 2)  # Only valid elements counted
        
        id1, _ = self.heap.pop_min()
        id2, _ = self.heap.pop_min()
        
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 3)

class TestHNSW(unittest.TestCase):
    """Test HNSW graph structure."""
    
    def setUp(self):
        self.hnsw = HNSW(M=5)  # Small M for testing
        
    def test_level_generation(self):
        """Test random level generation matches FAISS distribution."""
        levels = [self.hnsw.random_level() for _ in range(10000)]
        
        # Check level distribution properties
        self.assertTrue(max(levels) < 32)  # Should not exceed max level
        self.assertTrue(min(levels) >= 0)  # Should not be negative
        
        # Count levels
        level_counts = np.bincount(levels)
        
        # Check exponential distribution property
        for i in range(1, len(level_counts)-1):
            ratio = level_counts[i] / level_counts[i-1]
            expected_ratio = np.exp(-1 / self.hnsw.level_mult)
            self.assertAlmostEqual(ratio, expected_ratio, delta=0.1)
            
    def test_neighbor_selection(self):
        """Test neighbor selection with diversity heuristic."""
        # Create simple distance computer
        def dist_computer(a: int, b: int) -> float:
            return abs(a - b)
            
        # Add vertices in sequence
        self.hnsw.add_vertex(0, 1, dist_computer)
        self.hnsw.add_vertex(1, 1, dist_computer)
        self.hnsw.add_vertex(2, 1, dist_computer)
        
        # Check connections
        neighbors_0 = self.hnsw.get_neighbors(0, 0)
        neighbors_1 = self.hnsw.get_neighbors(1, 0)
        neighbors_2 = self.hnsw.get_neighbors(2, 0)
        
        # Verify diversity - vertices should connect to both close and far neighbors
        self.assertTrue(1 in neighbors_0)  # Close neighbor
        self.assertTrue(2 in neighbors_0)  # Far neighbor
        self.assertTrue(0 in neighbors_1 and 2 in neighbors_1)  # Both sides
        self.assertTrue(1 in neighbors_2)  # Close neighbor
        
    def test_search(self):
        """Test search functionality."""
        # Create simple distance computer
        def dist_computer(a: int, b: int) -> float:
            return abs(a - b)
            
        # Add vertices with different levels
        for i in range(10):
            level = 1 if i % 3 == 0 else 0  # Some vertices at higher level
            self.hnsw.add_vertex(i, level, dist_computer)
            
        # Search from different points
        for query in [0, 4, 9]:
            results = self.hnsw.search(query, ef=3, dist_computer=dist_computer)
            
            # Check results are sorted by distance
            self.assertTrue(all(d1 <= d2 for (d1, _), (d2, _) in zip(results, results[1:])))
            
            # Check closest points are found
            found_ids = [idx for _, idx in results]
            expected = sorted(range(10), key=lambda x: abs(x - query))[:3]
            self.assertEqual(set(found_ids), set(expected))

class TestHNSWIndex(unittest.TestCase):
    """Test HNSW index implementation."""
    
    def setUp(self):
        # Create synthetic test data
        np.random.seed(42)
        self.d = 16
        self.n_train = 1000
        self.n_test = 10
        self.k = 5
        
        # Generate random vectors
        self.train_data = np.random.randn(self.n_train, self.d).astype(np.float32)
        self.test_data = np.random.randn(self.n_test, self.d).astype(np.float32)
        
        # Create index
        self.index = HNSWFlatIndex(d=self.d, M=16)
        
    def test_add_search(self):
        """Test adding vectors and searching."""
        # Add vectors
        self.index.add(self.train_data.tolist())
        self.assertEqual(self.index.ntotal, self.n_train)
        
        # Search
        result = self.index.search(self.test_data.tolist(), self.k)
        
        # Check result format
        self.assertEqual(len(result.distances), self.n_test)
        self.assertEqual(len(result.labels), self.n_test)
        self.assertEqual(len(result.distances[0]), self.k)
        self.assertEqual(len(result.labels[0]), self.k)
        
        # Check distances are sorted
        for dists in result.distances:
            self.assertEqual(dists, sorted(dists))
            
        # Check labels are valid
        for labels in result.labels:
            self.assertTrue(all(0 <= l < self.n_train for l in labels))
            
    def test_reconstruction(self):
        """Test vector reconstruction."""
        # Add vectors
        self.index.add(self.train_data.tolist())
        
        # Reconstruct some vectors
        for i in range(10):
            reconstructed = self.index.reconstruct(i)
            original = self.train_data[i]
            
            # Check reconstruction is exact
            np.testing.assert_array_almost_equal(reconstructed, original)
            
    def test_search_accuracy(self):
        """Test search accuracy against exact search."""
        # Add vectors
        self.index.add(self.train_data.tolist())
        
        # Compute exact distances
        query = self.test_data[0]
        exact_dists = np.sum((self.train_data - query) ** 2, axis=1)
        exact_indices = np.argsort(exact_dists)[:self.k]
        
        # HNSW search
        result = self.index.search([query.tolist()], self.k)
        hnsw_indices = result.labels[0]
        
        # Compare results - should have good recall
        recall = len(set(exact_indices) & set(hnsw_indices)) / self.k
        self.assertGreater(recall, 0.8)  # At least 80% recall
        
    def test_batch_computation(self):
        """Test batch distance computation."""
        self.index.add(self.train_data.tolist())
        
        # Single query
        query = mx.array(self.test_data[0], dtype=mx.float32)
        vectors = mx.array(self.train_data[:4], dtype=mx.float32)
        
        # Compute distances in batch
        batch_dists = self.index._compute_distances_batch(query, vectors, batch_size=4)
        
        # Compute distances individually
        individual_dists = []
        for i in range(4):
            if self.index.metric_type == MetricType.L2:
                diff = vectors[i] - query
                dist = float(mx.sum(diff * diff))
            else:
                dist = -float(mx.dot(vectors[i], query))
            individual_dists.append(dist)
            
        # Compare results
        np.testing.assert_array_almost_equal(
            batch_dists.numpy(),
            np.array(individual_dists)
        )

if __name__ == '__main__':
    unittest.main()