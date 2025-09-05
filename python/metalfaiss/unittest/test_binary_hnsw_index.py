"""
test_binary_hnsw_index.py - Tests for binary HNSW index

These tests verify the functionality of BinaryHNSWIndex, particularly
its graph construction and search behavior in Hamming space.
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Tuple
from ..index.binary_hnsw_index import BinaryHNSWIndex

def generate_binary_vectors(n: int, d: int, seed: int = 42) -> List[List[int]]:
    """Generate random binary vectors."""
    np.random.seed(seed)
    return np.random.randint(0, 2, (n, d)).tolist()

def hamming_distance(x: List[int], y: List[int]) -> int:
    """Compute Hamming distance between binary vectors."""
    return sum(a != b for a, b in zip(x, y))

class TestBinaryHNSWIndex(unittest.TestCase):
    """Test binary HNSW index functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64  # Common binary vector size
        self.n = 1000  # Need more vectors for meaningful graph
        
        # Create vectors with some structure
        # Half the vectors are closer to 0s, half closer to 1s
        np.random.seed(42)
        n_half = self.n // 2
        zeros = np.random.binomial(1, 0.2, (n_half, self.d))  # Mostly zeros
        ones = np.random.binomial(1, 0.8, (n_half, self.d))   # Mostly ones
        self.vectors = np.vstack([zeros, ones]).tolist()
        
        # Create index with default parameters
        self.index = BinaryHNSWIndex(self.d)
        
    def test_initialization(self):
        """Test index initialization."""
        self.assertEqual(self.index.d, self.d)
        self.assertEqual(self.index.M, 32)  # Default value
        self.assertEqual(self.index.efConstruction, 40)  # Default value
        self.assertEqual(self.index.efSearch, 16)  # Default value
        self.assertEqual(self.index.ntotal, 0)
        self.assertTrue(self.index.is_trained)  # No training needed
        
        # Custom parameters
        index = BinaryHNSWIndex(
            self.d,
            M=64,
            efConstruction=100,
            efSearch=50
        )
        self.assertEqual(index.M, 64)
        self.assertEqual(index.efConstruction, 100)
        self.assertEqual(index.efSearch, 50)
        
        # Invalid parameters
        with self.assertRaises(ValueError):
            BinaryHNSWIndex(0)  # Invalid dimension
            
    def test_parameters(self):
        """Test parameter validation."""
        # Valid values
        self.index.efConstruction = 50
        self.assertEqual(self.index.efConstruction, 50)
        
        self.index.efSearch = 30
        self.assertEqual(self.index.efSearch, 30)
        
        # Invalid values
        with self.assertRaises(ValueError):
            self.index.efConstruction = 0
        with self.assertRaises(ValueError):
            self.index.efSearch = -1
            
    def test_adding(self):
        """Test vector addition."""
        # Add vectors
        self.index.add(self.vectors)
        self.assertEqual(self.index.ntotal, self.n)
        self.assertIsNotNone(self.index.xb)
        self.assertEqual(self.index.xb.shape, (self.n, self.d))
        
        # Verify HNSW graph
        self.assertGreater(self.index.hnsw.max_level, 0)
        self.assertGreaterEqual(self.index.hnsw.entry_point, 0)
        
        # Add with IDs
        more_vectors = generate_binary_vectors(10, self.d, seed=43)
        ids = list(range(100, 110))
        self.index.add(more_vectors, ids)
        
        self.assertEqual(self.index.ntotal, self.n + 10)
        
        # Verify IDs were stored
        vector = self.index.reconstruct(100)  # Should find first added vector
        self.assertEqual(len(vector), self.d)
        self.assertEqual(vector, more_vectors[0])
        
    def test_searching(self):
        """Test nearest neighbor search."""
        self.index.add(self.vectors)
        
        # Search with k=1
        query = self.vectors[:1]  # Should find itself
        result = self.index.search(query, k=1)
        
        self.assertEqual(len(result.distances), 1)
        self.assertEqual(len(result.labels), 1)
        self.assertEqual(result.distances[0][0], 0)  # Exact match
        
        # Search with k=5 and different efSearch values
        query = generate_binary_vectors(1, self.d, seed=44)[0]
        k = 5
        
        for ef in [16, 32, 64, 128]:
            self.index.efSearch = ef
            result = self.index.search([query], k)
            
            # Basic checks
            self.assertEqual(len(result.distances[0]), k)
            self.assertTrue(all(d >= 0 for d in result.distances[0]))
            self.assertTrue(
                all(d1 <= d2 for d1, d2 in zip(
                    result.distances[0],
                    result.distances[0][1:]
                ))
            )  # Sorted distances
            
            # Higher ef should find same or better distances
            if ef > 16:
                prev_dist = prev_result.distances[0][0]
                curr_dist = result.distances[0][0]
                self.assertLessEqual(curr_dist, prev_dist)
                
            prev_result = result
            
        # Verify distances are Hamming
        result = self.index.search([query], k=10)
        for i, label in enumerate(result.labels[0]):
            if label != -1:
                dist = hamming_distance(query, self.vectors[label])
                self.assertEqual(dist, result.distances[0][i])
                
    def test_graph_quality(self):
        """Test HNSW graph structure quality."""
        # Add vectors with higher efConstruction for better quality
        self.index.efConstruction = 100
        self.index.add(self.vectors)
        
        # Test graph connectivity
        # Each vertex should connect to similar vectors
        for i in range(0, self.n, 100):  # Test subset
            # Get neighbors at level 0
            neighbors = self.index.hnsw.get_neighbors(i, 0)
            neighbors = [n for n in neighbors.tolist() if n != -1]
            
            if not neighbors:
                continue
                
            # Compute distances to all vectors
            dists = []
            for j in range(self.n):
                if j != i:
                    dist = hamming_distance(
                        self.vectors[i],
                        self.vectors[j]
                    )
                    dists.append((dist, j))
            dists.sort()
            
            # Neighbors should be among closest vectors
            neighbor_dists = set(
                hamming_distance(
                    self.vectors[i],
                    self.vectors[n]
                )
                for n in neighbors
            )
            closest_dists = set(d for d, _ in dists[:len(neighbors)])
            
            # Allow some difference but should be mostly close
            common = len(neighbor_dists & closest_dists)
            self.assertGreater(
                common / len(neighbors),
                0.5,  # At least 50% should be among closest
                f"Poor neighbor quality for vertex {i}"
            )
            
    def test_reconstruction(self):
        """Test vector reconstruction."""
        # Add with explicit IDs
        ids = list(range(100, 100 + self.n))
        self.index.add(self.vectors, ids)
        
        # Reconstruct with ID
        vector = self.index.reconstruct(100)  # First vector
        self.assertEqual(vector, self.vectors[0])
        
        # Invalid ID
        with self.assertRaises(ValueError):
            self.index.reconstruct(-1)
        with self.assertRaises(ValueError):
            self.index.reconstruct(200)  # Beyond added vectors
            
    def test_empty_index(self):
        """Test operations on empty index."""
        # Search should fail
        with self.assertRaises(RuntimeError):
            self.index.search(self.vectors[:1], k=1)
            
        # Reconstruction should fail
        with self.assertRaises(RuntimeError):
            self.index.reconstruct(0)
            
    def test_reset(self):
        """Test index reset."""
        self.index.add(self.vectors)
        
        self.assertEqual(self.index.ntotal, self.n)
        self.assertIsNotNone(self.index.xb)
        
        # Reset should clear everything
        self.index.reset()
        self.assertEqual(self.index.ntotal, 0)
        self.assertIsNone(self.index.xb)
        self.assertIsNone(self.index.ids)
        
        # But parameters remain
        self.assertEqual(self.index.M, 32)
        self.assertEqual(self.index.efConstruction, 40)
        self.assertEqual(self.index.efSearch, 16)

if __name__ == '__main__':
    unittest.main()