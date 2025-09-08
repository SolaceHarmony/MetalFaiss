"""
test_product_quantizer.py - Tests for Product Quantizer implementation
"""

import unittest
import mlx.core as mx
from ..utils.rng_utils import new_key
from ..types.metric_type import MetricType
from ..index.product_quantizer_index import ProductQuantizer, ProductQuantizerIndex

class TestProductQuantizer(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        k = new_key(42)
        self.d = 16  # Dimension must be divisible by M
        self.M = 4   # Number of subquantizers
        self.nbits = 4  # 16 centroids per subquantizer
        self.n_train = 1000
        self.n_test = 100
        
        # Training data
        self.train_data = mx.random.normal(shape=(self.n_train, self.d), key=k).astype(mx.float32)
        
        # Test data
        self.test_data = mx.random.normal(shape=(self.n_test, self.d), key=k).astype(mx.float32)
        
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
        k = new_key(42)
        self.d = 16
        self.M = 4
        self.nbits = 4
        self.n_train = 1000
        self.n_database = 500
        self.n_query = 10
        self.k = 5
        
        # Training data
        self.train_data = mx.random.normal(shape=(self.n_train, self.d), key=k).astype(mx.float32)
        
        # Database vectors
        self.database = mx.random.normal(shape=(self.n_database, self.d), key=k).astype(mx.float32)
        
        # Query vectors
        self.queries = mx.random.normal(shape=(self.n_query, self.d), key=k).astype(mx.float32)
        
        # Initialize index
        self.index = ProductQuantizerIndex(self.d, self.M, self.nbits)
        
    def test_training_and_adding(self):
        """Test index training and vector addition."""
        # Train
        self.index.train(self.train_data)
        self.assertTrue(self.index.is_trained)
        
        # Add vectors
        self.index.add(self.database)
        self.assertEqual(self.index.ntotal, self.n_database)
        
    def test_searching(self):
        """Test nearest neighbor search."""
        # Train and add vectors
        self.index.train(self.train_data.tolist())
        self.index.add(self.database.tolist())
        
        # Search
        result = self.index.search(self.queries, self.k)
        
        # Check result shapes
        self.assertEqual(int(result.distances.shape[0]), self.n_query)
        self.assertEqual(int(result.indices.shape[0]), self.n_query)
        self.assertEqual(int(result.distances.shape[1]), self.k)
        self.assertEqual(int(result.indices.shape[1]), self.k)
        
        # Verify distances are sorted
        for i in range(self.n_query):
            drow = result.distances[i]
            self.assertTrue(bool(mx.all(mx.less_equal(drow[:-1], drow[1:])).item()))
            
        # Verify labels are valid
        for i in range(self.n_query):
            lrow = result.indices[i]
            self.assertTrue(bool(mx.all(mx.logical_and(mx.greater_equal(lrow, 0), mx.less(lrow, self.n_database))).item()))
            
    def test_reconstruction(self):
        """Test vector reconstruction."""
        # Train and add vectors
        self.index.train(self.train_data)
        self.index.add(self.database)
        
        # Reconstruct a vector
        key = 0
        reconstructed = self.index.reconstruct(key)
        
        # Check shape
        self.assertEqual(int(reconstructed.shape[0]), 1 if len(reconstructed.shape) > 1 else self.d)
        
        # Verify reconstruction error
        original = self.database[key]
        rec = reconstructed if len(reconstructed.shape) == 1 else reconstructed[0]
        mse = float(mx.mean(mx.square(mx.subtract(rec, original))).item())
        self.assertLess(mse, 1.0)  # Adjust threshold as needed

if __name__ == '__main__':
    unittest.main()
