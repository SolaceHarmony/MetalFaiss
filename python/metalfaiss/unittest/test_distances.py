# unittest/test_distances.py
import unittest
import mlx.core as mx
import numpy as np
from ..metric_type import MetricType
from faissmlx.distances import pairwise_L2sqr  # assuming this is our MLX wrapper
from faissmlx.extra_distances import pairwise_extra_distances, knn_extra_metrics

class TestDistances(unittest.TestCase):
    def setUp(self):
        # Create some synthetic data for testing:
        # For instance, 10 query vectors and 20 database vectors of dimension 5.
        self.nq = 10
        self.nb = 20
        self.d = 5
        
        # Use MLX to create arrays. Under the hood, these are NumPy arrays.
        np.random.seed(42)
        self.xq = mx.array(np.random.randn(self.nq, self.d).astype(np.float32))
        self.xb = mx.array(np.random.randn(self.nb, self.d).astype(np.float32))
    
    def test_pairwise_L2sqr(self):
        # Compute L2 squared distances using our function.
        dis = pairwise_L2sqr(self.xq, self.xb)
        # Verify shape is (nq, nb)
        self.assertEqual(dis.shape, (self.nq, self.nb))
        # Optionally, compare with a NumPy baseline computation
        xq_np = self.xq.numpy()
        xb_np = self.xb.numpy()
        expected = np.sum((xq_np[:, None, :] - xb_np[None, :, :]) ** 2, axis=2)
        np.testing.assert_allclose(dis.numpy(), expected, rtol=1e-5)
    
    def test_extra_distances_L1(self):
        # Test extra_distances with L1 metric.
        dis = pairwise_extra_distances(self.xq, self.xb, MetricType.L1)
        self.assertEqual(dis.shape, (self.nq, self.nb))
        xq_np = self.xq.numpy()
        xb_np = self.xb.numpy()
        expected = np.sum(np.abs(xq_np[:, None, :] - xb_np[None, :, :]), axis=2)
        np.testing.assert_allclose(dis.numpy(), expected, rtol=1e-5)
    
    def test_knn_extra_metrics(self):
        # Test kNN search with extra metrics.
        k = 3
        values, indices = knn_extra_metrics(self.xq, self.xb, MetricType.L2, k)
        self.assertEqual(values.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))
        # For L2, the neighbors with smallest distances are returned.
        # Here, we can at least check that the indices are in a valid range.
        self.assertTrue(np.all((indices.numpy() >= 0) & (indices.numpy() < self.nb)))

if __name__ == '__main__':
    unittest.main()