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
from ..distances import (
    fvec_canberra,
    fvec_bray_curtis,
    fvec_jensen_shannon,
    pairwise_extra_distances,
    pairwise_jaccard,
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
        
    def test_canberra_distance(self):
        """Test Canberra distance computation."""
        x = self.xq[0]
        y = self.xb[0]
        
        # MLX implementation
        dist = fvec_canberra(x, y)
        
        # NumPy reference
        x_np = np.array(x.tolist()); y_np = np.array(y.tolist())
        num = np.abs(x_np - y_np)
        den = np.abs(x_np) + np.abs(y_np)
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
        x_np = np.array(x.tolist()); y_np = np.array(y.tolist())
        num = np.sum(np.abs(x_np - y_np))
        den = np.sum(np.abs(x_np + y_np))
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
        self.assertFalse(np.isnan(float(dist)))
        self.assertGreaterEqual(dist, 0.0)
        
    def test_pairwise_distances(self):
        """Test pairwise distance computations."""
        # Test each metric type
        for metric in ["Canberra", "BrayCurtis", "JensenShannon"]:
            distances = pairwise_extra_distances(
                self.xq[:2],
                self.xb[:3],
                metric
            )
            
            # Check shape
            self.assertEqual(distances.shape, (2, 3))
            
            # Verify distances are non-negative for distance metrics
            self.assertTrue(mx.all(distances >= 0))
    def test_pairwise_jaccard(self):
        # Non-negative vectors only
        dists = pairwise_jaccard(self.xq_pos[:2], self.xb_pos[:3])
        self.assertEqual(dists.shape, (2, 3))
        self.assertTrue(mx.all(dists >= 0))

if __name__ == '__main__':
    unittest.main()
