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
from ..types.metric_type import MetricType
from ..distances import (
    fvec_canberra,
    fvec_bray_curtis,
    fvec_jensen_shannon,
    pairwise_extra_distances,
    pairwise_jaccard,
)
from ..utils.rng_utils import new_key, split2

class TestExtraDistances(unittest.TestCase):
    """Test extra distance metrics."""
    
    def setUp(self):
        """Create test vectors."""
        self.d = 16  # Dimension
        self.nq = 10  # Number of queries
        self.nb = 20  # Number of database vectors

        k = new_key(42)
        kq, kb = split2(k)
        # Create random vectors
        self.xq = mx.random.normal(shape=(self.nq, self.d), key=kq).astype(mx.float32)
        self.xb = mx.random.normal(shape=(self.nb, self.d), key=kb).astype(mx.float32)

        # Create non-negative vectors for Jaccard distance
        self.xq_pos = mx.abs(self.xq)
        self.xb_pos = mx.abs(self.xb)
        
    def test_canberra_distance(self):
        """Test Canberra distance computation."""
        x = self.xq[0]
        y = self.xb[0]
        
        # MLX implementation
        dist = fvec_canberra(x, y)

        # MLX reference
        num = mx.abs(mx.subtract(x, y))
        den = mx.add(mx.abs(x), mx.abs(y))
        den = mx.where(mx.greater(den, mx.zeros_like(den)), den, mx.ones_like(den))
        ref = mx.sum(mx.divide(num, den))

        self.assertTrue(bool(mx.allclose(dist, ref, rtol=1e-5, atol=1e-8)))
        
    def test_bray_curtis_distance(self):
        """Test Bray-Curtis distance computation."""
        x = self.xq[0]
        y = self.xb[0]
        
        # MLX implementation
        dist = fvec_bray_curtis(x, y)

        # MLX reference
        num = mx.sum(mx.abs(mx.subtract(x, y)))
        den = mx.sum(mx.abs(mx.add(x, y)))
        ref = mx.divide(num, mx.maximum(den, mx.array(1e-20, dtype=num.dtype)))

        self.assertTrue(bool(mx.allclose(dist, ref, rtol=1e-5, atol=1e-8)))
        
    def test_jensen_shannon_stability(self):
        """Test Jensen-Shannon distance numerical stability."""
        # Test with vectors containing zeros
        x = mx.array([0.0, 0.1, 0.2, 0.7])
        y = mx.array([0.1, 0.0, 0.3, 0.6])
        
        # MLX implementation
        dist = fvec_jensen_shannon(x, y)

        # Should not produce NaN
        self.assertFalse(bool(mx.isnan(dist).item()))  # boundary-ok
        self.assertTrue(bool(mx.greater_equal(dist, mx.array(0.0, dtype=dist.dtype)).item()))
        
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
            self.assertTrue(bool(mx.all(mx.greater_equal(distances, mx.zeros_like(distances))).item()))
    def test_pairwise_jaccard(self):
        # Non-negative vectors only
        dists = pairwise_jaccard(self.xq_pos[:2], self.xb_pos[:3])
        self.assertEqual(dists.shape, (2, 3))
        self.assertTrue(bool(mx.all(mx.greater_equal(dists, mx.zeros_like(dists))).item()))

if __name__ == '__main__':
    unittest.main()
