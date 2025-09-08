"""
Correctness checks for top-k SVD via subspace iteration.

Verifies orthonormality and reconstruction on random matrices.
"""

import unittest
import mlx.core as mx

from ..faissmlx.svd import topk_svd


class TestSVDCorrectness(unittest.TestCase):
    def test_svd_shapes_and_orthogonality(self):
        m, n, k = 256, 128, 16
        A = mx.random.normal(shape=(m, n)).astype(mx.float32)

        U, S, Vt = topk_svd(A, k, iters=5, use_kernel=False)
        mx.eval(U, S, Vt)
        self.assertEqual(U.shape, (m, k))
        self.assertEqual(S.shape, (k,))
        self.assertEqual(Vt.shape, (k, n))

        # Orthonormal columns in U and rows in Vt
        UtU = mx.matmul(mx.transpose(U), U)
        VVt = mx.matmul(Vt, mx.transpose(Vt))
        # V rows should be orthonormal (within modest tolerance)
        self.assertTrue(bool(mx.allclose(VVt, mx.eye(k), rtol=1e-2, atol=1e-2)))
        # U columns are normalized; allow looser tolerance for orthogonality
        diag_err = mx.max(mx.abs(mx.diag(UtU) - 1.0))
        off = UtU - mx.diag(mx.diag(UtU))
        off_max = mx.max(mx.abs(off))
        self.assertTrue(bool(mx.all(mx.less_equal(diag_err, mx.array(1e-1, dtype=mx.float32))).item()))  # boundary-ok
        self.assertTrue(bool(mx.all(mx.less_equal(off_max, mx.array(2e-1, dtype=mx.float32))).item()))  # boundary-ok

    def test_svd_reconstruction_quality(self):
        m, n, k = 256, 128, 16
        A = mx.random.normal(shape=(m, n)).astype(mx.float32)
        U, S, Vt = topk_svd(A, k, iters=4, use_kernel=False)
        mx.eval(U, S, Vt)
        # Check ||A V - U S||_F / ||A||_F is small
        V = mx.transpose(Vt)
        AV = mx.matmul(A, V)
        US = U * S.reshape((1, -1))
        num = mx.sqrt(mx.sum((AV - US) * (AV - US)))
        den = mx.sqrt(mx.sum(A * A)) + 1e-6
        rel = num / den
        # For random matrices and a few iters, relative residual should be modest
        self.assertTrue(bool(mx.all(mx.less_equal(rel, mx.array(0.5, dtype=mx.float32))).item()))  # boundary-ok

if __name__ == "__main__":
    unittest.main()
