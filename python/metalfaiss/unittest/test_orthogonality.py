"""
Tests for orthogonality helpers: columns/rows, completion, initializer, blocked.
"""

import unittest
import mlx.core as mx

from ..faissmlx.orthogonality import (
    orthonormal_columns,
    orthonormal_rows,
    complete_basis,
    orthogonal,
    orthogonalize_blocked,
)


def _orth_err_cols(Q: mx.array) -> float:
    m, n = int(Q.shape[0]), int(Q.shape[1])
    QtQ = mx.matmul(mx.transpose(Q), Q)
    I = mx.eye(n)
    mx.eval(QtQ, I)
    return float(mx.max(mx.abs(QtQ - I)).item())  # boundary-ok


def _orth_err_rows(Q: mx.array) -> float:
    m, n = int(Q.shape[0]), int(Q.shape[1])
    QQT = mx.matmul(Q, mx.transpose(Q))
    I = mx.eye(m)
    mx.eval(QQT, I)
    return float(mx.max(mx.abs(QQT - I)).item())  # boundary-ok


class TestOrthogonality(unittest.TestCase):
    def test_orthonormal_columns(self):
        mx.random.seed(0)
        X = mx.random.normal(shape=(64, 32)).astype(mx.float32)
        Q = orthonormal_columns(X)
        err = _orth_err_cols(Q)
        self.assertLess(err, 1e-3)

    def test_orthonormal_rows(self):
        mx.random.seed(1)
        X = mx.random.normal(shape=(32, 64)).astype(mx.float32)
        Q = orthonormal_rows(X)
        err = _orth_err_rows(Q)
        self.assertLess(err, 1e-3)

    def test_complete_basis(self):
        mx.random.seed(2)
        X = mx.random.normal(shape=(48, 16)).astype(mx.float32)
        Q = orthonormal_columns(X)
        R = complete_basis(Q)
        self.assertEqual(R.shape, (48, 48))
        err = _orth_err_cols(R)
        self.assertLess(err, 1e-3)

    def test_orthogonal_initializer(self):
        mx.random.seed(3)
        W = orthogonal((64, 32))
        err = _orth_err_cols(W)
        self.assertLess(err, 1e-3)

    def test_orthogonalize_blocked(self):
        mx.random.seed(4)
        X = mx.random.normal(shape=(64, 40)).astype(mx.float32)
        Q = orthogonalize_blocked(X, B=16)
        err = _orth_err_cols(Q)
        self.assertLess(err, 5e-3)  # slightly looser tolerance


if __name__ == "__main__":
    unittest.main()
