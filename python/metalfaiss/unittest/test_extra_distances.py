"""
test_extra_distances.py - Tests for extra distance metrics

These tests verify that our MLX implementations of extra distance metrics match
the behavior of the original FAISS implementations, particularly around:
- Numerical stability
- Edge cases
- Batch operations
"""

import unittest
import mlx.core as mx
import numpy as np
from ..types.metric_type import MetricType
from ..faissmlx.extra_distances import (
    # Vector distance functions
    fvec_Lp,
    fvec_canberra,
    fvec_bray_curtis,
    fvec_jensen_shannon,
    fvec_jaccard,
    fvec_nan_euclidean,
    fvec_abs_inner_product,
    
    # Distance classes
    L2Distance,
    InnerProductDistance,
    LpDistance,
    AbsInnerProductDistance,
    
    # Batch operations
    pairwise_extra_distances,
    knn_extra_metrics
)

class TestExtraDistances(unittest.TestCase):
    """Test extra distance metrics."""
    
    def setUp(self):
        """Create test vectors."""
        np.random.seed(42)
        self.d = 16  # Dimension
        self.nq = 10  # Number of queries
        self.nb = 20  # Number of database vectors
        
        # Create random vectors
        self.xq = mx.array(np.random.randn(self.nq, self.d).astype(np.float32))
        self.xb = mx.array(np.random.randn(self.nb, self.d).astype(np.float32))
        
        # Create non-negative vectors for Jaccard distance
        self.xq_pos = mx.array(np.abs(self.xq))
        self.xb_pos = mx.array(np.abs(self.xb))
        
    def test_lp_distance(self):
        """Test Lp distance computation."""
        x = self.xq[0]
        y = self.xb[0]
        p = 3.0  # Test with p=3
        
        # MLX implementation
        dist = fvec_Lp(x, y, p)
        
        # NumPy reference
        np_dist = np.sum(np.abs(x.numpy() - y.numpy()) ** p)
        
        self.assertAlmostEqual(dist, np_dist, places=5)
        
    def test_canberra_distance(self):
        """Test Canberra distance computation."""
        x = self.xq[0]
        y = self.xb[0]
        
        # MLX implementation
        dist = fvec_canberra(x, y)
        
        # NumPy reference
        num = np.abs(x.numpy() - y.numpy())
        den = np.abs(x.numpy()) + np.abs(y.numpy())
        den[den == 0] = 1.0  # Avoid division by zero
        np_dist = np.sum(num / den)
        
        self.assertAlmostEqual(dist, np_dist, places=5)
        
    def test_bray_curtis_distance(self):
        """Test Bray-Curtis distance computation."""
        x = self.xq[0]
        y = self.xb[0]
        
        # MLX implementation
        dist = fvec_bray_curtis(x, y)
        
        # NumPy reference
        num = np.sum(np.abs(x.numpy() - y.numpy()))
        den = np.sum(np.abs(x.numpy() + y.numpy()))
        np_dist = num / max(den, 1e-20)
        
        self.assertAlmostEqual(dist, np_dist, places=5)
        
    def test_jensen_shannon_stability(self):
        """Test Jensen-Shannon distance numerical stability."""
        # Test with vectors containing zeros
        x = mx.array([0.0, 0.1, 0.2, 0.7])
        y = mx.array([0.1, 0.0, 0.3, 0.6])
        
        # MLX implementation
        dist = fvec_jensen_shannon(x, y)
        
        # Should not produce NaN
        self.assertFalse(np.isnan(dist))
        self.assertGreaterEqual(dist, 0.0)
        
    def test_jaccard_validation(self):
        """Test Jaccard distance input validation."""
        # Should work with non-negative vectors
        dist = fvec_jaccard(self.xq_pos[0], self.xb_pos[0])
        self.assertGreaterEqual(dist, 0.0)
        self.assertLessEqual(dist, 1.0)
        
        # Should raise error with negative values
        with self.assertRaises(ValueError):
            fvec_jaccard(self.xq[0], self.xb[0])
            
    def test_nan_euclidean(self):
        """Test NaN-Euclidean distance."""
        # Create vectors with NaN values
        x = mx.array([1.0, float('nan'), 3.0, 4.0])
        y = mx.array([1.0, 2.0, float('nan'), 4.0])
        
        # MLX implementation
        dist = fvec_nan_euclidean(x, y)
        
        # Only two valid pairs: indices 0 and 3
        # Scale factor should be 4/2 = 2
        expected = 2.0 * ((1.0 - 1.0)**2 + (4.0 - 4.0)**2)
        self.assertAlmostEqual(dist, expected, places=5)
        
    def test_abs_inner_product(self):
        """Test absolute inner product."""
        x = self.xq[0]
        y = self.xb[0]
        
        # MLX implementation
        dist = fvec_abs_inner_product(x, y)
        
        # NumPy reference
        np_dist = np.sum(np.abs(x.numpy() * y.numpy()))
        
        self.assertAlmostEqual(dist, np_dist, places=5)
        
    def test_distance_classes(self):
        """Test distance class implementations."""
        x = self.xq[0]
        y = self.xb[0]
        
        # L2 distance
        l2_dist = L2Distance(self.d)
        self.assertFalse(l2_dist.is_similarity)
        self.assertAlmostEqual(
            l2_dist(x, y),
            np.sum((x.numpy() - y.numpy())**2),
            places=5
        )
        
        # Inner product
        ip_dist = InnerProductDistance(self.d)
        self.assertTrue(ip_dist.is_similarity)
        self.assertAlmostEqual(
            ip_dist(x, y),
            np.dot(x.numpy(), y.numpy()),
            places=5
        )
        
        # Lp distance
        p = 3.0
        lp_dist = LpDistance(self.d, p)
        self.assertFalse(lp_dist.is_similarity)
        self.assertAlmostEqual(
            lp_dist(x, y),
            np.sum(np.abs(x.numpy() - y.numpy())**p),
            places=5
        )
        
        # Absolute inner product
        abs_ip_dist = AbsInnerProductDistance(self.d)
        self.assertTrue(abs_ip_dist.is_similarity)
        self.assertAlmostEqual(
            abs_ip_dist(x, y),
            np.sum(np.abs(x.numpy() * y.numpy())),
            places=5
        )
        
    def test_pairwise_distances(self):
        """Test pairwise distance computations."""
        # Test each metric type
        for metric_type in [
            MetricType.L2,
            MetricType.INNER_PRODUCT,
            MetricType.L1,
            MetricType.LINF,
            MetricType.LP,
            MetricType.CANBERRA,
            MetricType.BRAY_CURTIS,
            MetricType.JENSEN_SHANNON,
            MetricType.ABS_INNER_PRODUCT
        ]:
            # Skip Jaccard test for now (needs non-negative inputs)
            if metric_type == MetricType.JACCARD:
                continue
                
            # Compute distances
            distances = pairwise_extra_distances(
                self.xq[:2],  # Use small batch for testing
                self.xb[:3],
                metric_type
            )
            
            # Check shape
            self.assertEqual(distances.shape, (2, 3))
            
            # Verify distances are non-negative for distance metrics
            if metric_type not in {
                MetricType.INNER_PRODUCT,
                MetricType.ABS_INNER_PRODUCT
            }:
                self.assertTrue(mx.all(distances >= 0))
                
    def test_knn_search(self):
        """Test k-nearest neighbor search."""
        k = 5
        
        # Test each metric type
        for metric_type in [
            MetricType.L2,
            MetricType.INNER_PRODUCT,
            MetricType.L1,
            MetricType.LINF,
            MetricType.ABS_INNER_PRODUCT
        ]:
            # Search
            distances, indices = knn_extra_metrics(
                self.xq,
                self.xb,
                metric_type,
                k
            )
            
            # Check shapes
            self.assertEqual(distances.shape, (self.nq, k))
            self.assertEqual(indices.shape, (self.nq, k))
            
            # Verify indices are valid
            self.assertTrue(mx.all(indices >= 0))
            self.assertTrue(mx.all(indices < self.nb))
            
            # Verify distances are sorted
            if metric_type in {
                MetricType.INNER_PRODUCT,
                MetricType.ABS_INNER_PRODUCT
            }:
                # Similarity metrics: largest first
                self.assertTrue(
                    all(mx.all(distances[i][:-1] >= distances[i][1:])
                        for i in range(len(distances)))
                )
            else:
                # Distance metrics: smallest first
                self.assertTrue(
                    all(mx.all(distances[i][:-1] <= distances[i][1:])
                        for i in range(len(distances)))
                )

if __name__ == '__main__':
    unittest.main()