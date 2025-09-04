# unittest/test_distances.py
import unittest
import numpy as np

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    # Use numpy as fallback
    class MockMLX:
        @staticmethod
        def array(data, dtype=None):
            return np.array(data, dtype=dtype)
        
        @staticmethod  
        def eval(x):
            return x
    
    mx = MockMLX()

from ..metric_type import MetricType

try:
    from ..distances import pairwise_L2sqr
    from ..extra_distances import pairwise_extra_distances, knn_extra_metrics
    _HAS_DISTANCES = True
except ImportError:
    _HAS_DISTANCES = False

class TestDistances(unittest.TestCase):
    def setUp(self):
        if not _HAS_DISTANCES:
            self.skipTest("Distance functions not available")
            
        # Create some synthetic data for testing:
        # For instance, 10 query vectors and 20 database vectors of dimension 5.
        self.nq = 10
        self.nb = 20
        self.d = 5
        
        # Use numpy to create arrays and convert to MLX if available
        np.random.seed(42)
        self.xq = mx.array(np.random.randn(self.nq, self.d).astype(np.float32))
        self.xb = mx.array(np.random.randn(self.nb, self.d).astype(np.float32))
    
    def test_pairwise_L2sqr(self):
        # Compute L2 squared distances using our function.
        dis = pairwise_L2sqr(self.xq, self.xb)
        # Verify shape is (nq, nb)
        self.assertEqual(dis.shape, (self.nq, self.nb))
        # Optionally, compare with a NumPy baseline computation
        xq_np = np.array(self.xq)
        xb_np = np.array(self.xb)
        expected = np.sum((xq_np[:, None, :] - xb_np[None, :, :]) ** 2, axis=2)
        np.testing.assert_allclose(np.array(dis), expected, rtol=1e-5)
    
    def test_extra_distances_L1(self):
        # Test extra_distances with L1 metric.
        dis = pairwise_extra_distances(self.xq, self.xb, MetricType.L1)
        self.assertEqual(dis.shape, (self.nq, self.nb))
        xq_np = np.array(self.xq)
        xb_np = np.array(self.xb)
        expected = np.sum(np.abs(xq_np[:, None, :] - xb_np[None, :, :]), axis=2)
        np.testing.assert_allclose(np.array(dis), expected, rtol=1e-5)
    
    def test_knn_extra_metrics(self):
        # Test kNN search with extra metrics.
        k = 3
        values, indices = knn_extra_metrics(self.xq, self.xb, MetricType.L2, k)
        self.assertEqual(values.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))
        # For L2, the neighbors with smallest distances are returned.
        # Here, we can at least check that the indices are in a valid range.
        self.assertTrue(np.all((np.array(indices) >= 0) & (np.array(indices) < self.nb)))

if __name__ == '__main__':
    unittest.main()