"""
test_binary_ivf_index.py - Tests for binary IVF index

These tests verify the functionality of BinaryIVFIndex, particularly
its inverted list structure and search behavior.
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import List, Tuple
from ..index.binary_ivf_index import BinaryIVFIndex
from ..index.binary_flat_index import BinaryFlatIndex

def generate_binary_vectors(n: int, d: int, seed: int = 42) -> List[List[int]]:
    """Generate random binary vectors."""
    np.random.seed(seed)
    return np.random.randint(0, 2, (n, d)).tolist()

def hamming_distance(x: List[int], y: List[int]) -> int:
    """Compute Hamming distance between binary vectors."""
    return sum(a != b for a, b in zip(x, y))

class TestBinaryIVFIndex(unittest.TestCase):
    """Test binary IVF index functionality."""
    
    def setUp(self):
        """Create test data."""
        self.d = 64  # Common binary vector size
        self.nlist = 10  # Number of inverted lists
        self.n = 1000  # More vectors needed for meaningful clustering
        
        # Create vectors with some structure
        # Half the vectors are closer to 0s, half closer to 1s
        np.random.seed(42)
        n_half = self.n // 2
        zeros = np.random.binomial(1, 0.2, (n_half, self.d))  # Mostly zeros
        ones = np.random.binomial(1, 0.8, (n_half, self.d))   # Mostly ones
        self.vectors = np.vstack([zeros, ones]).tolist()
        
        # Create quantizer and index
        self.quantizer = BinaryFlatIndex(self.d)
        self.index = BinaryIVFIndex(self.quantizer, self.d, self.nlist)
        
    def test_initialization(self):
        """Test index initialization."""
        self.assertEqual(self.index.d, self.d)
        self.assertEqual(self.index.nlist, self.nlist)
        self.assertEqual(self.index.nprobe, 1)  # Default value
        self.assertEqual(self.index.ntotal, 0)
        self.assertFalse(self.index.is_trained)
        
        # Invalid dimensions
        with self.assertRaises(ValueError):
            wrong_quantizer = BinaryFlatIndex(self.d + 1)
            BinaryIVFIndex(wrong_quantizer, self.d, self.nlist)
            
        # Invalid nlist
        with self.assertRaises(ValueError):
            BinaryIVFIndex(self.quantizer, self.d, 0)
            
    def test_nprobe(self):
        """Test nprobe parameter."""
        # Valid values
        self.index.nprobe = 1
        self.assertEqual(self.index.nprobe, 1)
        
        self.index.nprobe = self.nlist
        self.assertEqual(self.index.nprobe, self.nlist)
        
        # Invalid values
        with self.assertRaises(ValueError):
            self.index.nprobe = 0
        with self.assertRaises(ValueError):
            self.index.nprobe = -1
            
    def test_training(self):
        """Test index training."""
        # Train with vectors
        self.index.train(self.vectors)
        
        self.assertTrue(self.index.is_trained)
        self.assertTrue(self.quantizer.is_trained)
        
        # Empty training data
        empty_index = BinaryIVFIndex(BinaryFlatIndex(self.d), self.d, self.nlist)
        with self.assertRaises(ValueError):
            empty_index.train([])
            
    def test_adding(self):
        """Test vector addition."""
        self.index.train(self.vectors)
        
        # Add vectors
        self.index.add(self.vectors)
        self.assertEqual(self.index.ntotal, self.n)
        
        # Check inverted lists
        list_sizes = [len(lst) for lst in self.index._invlists]
        self.assertEqual(sum(list_sizes), self.n)  # All vectors assigned
        self.assertTrue(all(s > 0 for s in list_sizes))  # No empty lists
        
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
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        
        # Search with k=1
        query = self.vectors[:1]  # Should find itself
        result = self.index.search(query, k=1)
        
        self.assertEqual(len(result.distances), 1)
        self.assertEqual(len(result.labels), 1)
        self.assertEqual(result.distances[0][0], 0)  # Exact match
        
        # Search with k=5 and different nprobe values
        query = generate_binary_vectors(1, self.d, seed=44)[0]
        k = 5
        
        for nprobe in [1, 2, 5, self.nlist]:
            self.index.nprobe = nprobe
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
            
            # More probe should find same or better distances
            if nprobe > 1:
                prev_dist = prev_result.distances[0][0]
                curr_dist = result.distances[0][0]
                self.assertLessEqual(curr_dist, prev_dist)
                
            prev_result = result
            
    def test_reconstruction(self):
        """Test vector reconstruction."""
        self.index.train(self.vectors)
        
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
        self.index.train(self.vectors)  # Trained but empty
        
        # Search should fail
        with self.assertRaises(RuntimeError):
            self.index.search(self.vectors[:1], k=1)
            
        # Reconstruction should fail
        with self.assertRaises(RuntimeError):
            self.index.reconstruct(0)
            
    def test_reset(self):
        """Test index reset."""
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        
        self.assertEqual(self.index.ntotal, self.n)
        
        # Reset should clear everything
        self.index.reset()
        self.assertEqual(self.index.ntotal, 0)
        self.assertTrue(all(len(lst) == 0 for lst in self.index._invlists))
        
        # But index remains trained
        self.assertTrue(self.index.is_trained)

if __name__ == '__main__':
    unittest.main()