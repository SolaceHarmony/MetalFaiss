import unittest
import mlx.core as mx

from ..types.metric_type import MetricType
from ..distances import pairwise_L2sqr, pairwise_L1

class TestDistances(unittest.TestCase):
    def setUp(self):
        # Create some synthetic data for testing:
        # For instance, 10 query vectors and 20 database vectors of dimension 5.
        self.nq = 10
        self.nb = 20
        self.d = 5
        
        # MLX-only random inputs
        self.xq = mx.random.normal(shape=(self.nq, self.d)).astype(mx.float32)
        self.xb = mx.random.normal(shape=(self.nb, self.d)).astype(mx.float32)
    
    def test_pairwise_L2sqr(self):
        # Compute L2 squared distances using our function.
        dis = pairwise_L2sqr(self.xq, self.xb)
        # Verify shape is (nq, nb)
        self.assertEqual(dis.shape, (self.nq, self.nb))
        # Compare with MLX baseline computation
        expected = mx.sum((self.xq[:, None, :] - self.xb[None, :, :]) * (self.xq[:, None, :] - self.xb[None, :, :]), axis=2)
        ok = mx.allclose(dis, expected, rtol=1e-5, atol=1e-8)
        self.assertTrue(bool(ok))
    
    def test_pairwise_L1(self):
        # Test L1 pairwise distances.
        dis = pairwise_L1(self.xq, self.xb)
        self.assertEqual(dis.shape, (self.nq, self.nb))
        expected = mx.sum(mx.abs(self.xq[:, None, :] - self.xb[None, :, :]), axis=2)
        ok = mx.allclose(dis, expected, rtol=1e-5, atol=1e-8)
        self.assertTrue(bool(ok))

if __name__ == '__main__':
    unittest.main()
