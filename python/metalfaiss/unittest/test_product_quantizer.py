"""
test_product_quantizer.py - Tests for Product Quantizer implementation
"""

import unittest
import mlx.core as mx
import numpy as np
from ..types.metric_type import MetricType
from ..index.product_quantizer_index import ProductQuantizer, ProductQuantizerIndex

class TestProductQuantizer(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        np.random.seed(42)
        self.d = 16  # Dimension must be divisible by M
        self.M = 4   # Number of subquantizers
        self.nbits = 4  # 16 centroids per subquantizer
        self.n_train = 1000
        self.n_test = 100
        
        # Training data
        self.train_data = mx.array(
            np.random.randn(self.n_train, self.d).astype(np.float32)
        )
        
        # Test data
        self.test_data = mx.array(
            np.random.randn(self.n_test, self.d).astype(np.float32)
        )
        
        # Initialize PQ
        self.pq = ProductQuantizer(self.d, self.M, self.nbits)
        
    def test_initialization(self):
        """Test PQ initialization."""
        self.assertEqual(self.pq.d, self.d)
        self.assertEqual(self.pq.M, self.M)
        self.assertEqual(self.pq.nbits, self.nbits)
        self.assertEqual(self.pq.dsub, self.d // self.M)
        self.assertEqual(self.pq.ksub, 1 << self.nbits)
        
    def test_training(self):
        """Test PQ training."""
        self.pq.train(self.train_data)
        
        # Check centroid shapes
        self.assertEqual(self.pq.centroids.shape, (self.M, self.pq.ksub, self.pq.dsub))
        self.assertEqual(self.pq.transposed_centroids.shape, (self.pq.dsub, self.M, self.pq.ksub))
        self.assertEqual(self.pq.centroids_sq_lengths.shape, (self.M, self.pq.ksub))
        
    def test_encode_decode(self):
        """Test encoding and decoding vectors."""
        self.pq.train(self.train_data)
        
        # Encode test vectors
        codes = self.pq.compute_codes(self.test_data)
        
        # Check code shapes and values
        self.assertEqual(codes.shape, (self.n_test, self.M))
        self.assertTrue(mx.all(codes >= 0))
        self.assertTrue(mx.all(codes < self.pq.ksub))
        
        # Decode and verify reconstruction error
        reconstructed = self.pq.decode(codes)
        self.assertEqual(reconstructed.shape, self.test_data.shape)
        
        # Reconstruction error should be reasonable
        mse = float(mx.mean((reconstructed - self.test_data) ** 2))
        self.assertLess(mse, 1.0)  # Adjust threshold as needed

class TestProductQuantizerIndex(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        np.random.seed(42)
        self.d = 16
        self.M = 4
        self.nbits = 4
        self.n_train = 1000
        self.n_database = 500
        self.n_query = 10
        self.k = 5
        
        # Training data
        self.train_data = np.random.randn(self.n_train, self.d).astype(np.float32)
        
        # Database vectors
        self.database = np.random.randn(self.n_database, self.d).astype(np.float32)
        
        # Query vectors
        self.queries = np.random.randn(self.n_query, self.d).astype(np.float32)
        
        # Initialize index
        self.index = ProductQuantizerIndex(self.d, self.M, self.nbits)
        
    def test_training_and_adding(self):
        """Test index training and vector addition."""
        # Train
        self.index.train(self.train_data.tolist())
        self.assertTrue(self.index.is_trained)
        
        # Add vectors
        self.index.add(self.database.tolist())
        self.assertEqual(self.index.ntotal, self.n_database)
        
    def test_searching(self):
        """Test nearest neighbor search."""
        # Train and add vectors
        self.index.train(self.train_data.tolist())
        self.index.add(self.database.tolist())
        
        # Search
        result = self.index.search(self.queries.tolist(), self.k)
        
        # Check result shapes
        self.assertEqual(len(result.distances), self.n_query)
        self.assertEqual(len(result.labels), self.n_query)
        self.assertEqual(len(result.distances[0]), self.k)
        self.assertEqual(len(result.labels[0]), self.k)
        
        # Verify distances are sorted
        for dists in result.distances:
            self.assertEqual(dists, sorted(dists))
            
        # Verify labels are valid
        for labels in result.labels:
            self.assertTrue(all(0 <= l < self.n_database for l in labels))
            
    def test_reconstruction(self):
        """Test vector reconstruction."""
        # Train and add vectors
        self.index.train(self.train_data.tolist())
        self.index.add(self.database.tolist())
        
        # Reconstruct a vector
        key = 0
        reconstructed = self.index.reconstruct(key)
        
        # Check shape
        self.assertEqual(len(reconstructed), self.d)
        
        # Verify reconstruction error
        original = self.database[key]
        mse = np.mean((np.array(reconstructed) - original) ** 2)
        self.assertLess(mse, 1.0)  # Adjust threshold as needed

if __name__ == '__main__':
    unittest.main()