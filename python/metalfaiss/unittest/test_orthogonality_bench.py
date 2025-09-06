"""
Micro-benchmarks for orthogonality helpers. Prints median times and errors.
"""

import time
import unittest
import mlx.core as mx

from ..faissmlx.orthogonality import (
    orthonormal_columns,
    orthogonalize_blocked,
    orthogonal,
)


def _median_time(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


def _orth_err_cols(Q: mx.array) -> float:
    m, n = int(Q.shape[0]), int(Q.shape[1])
    QtQ = mx.matmul(mx.transpose(Q), Q)
    I = mx.eye(n)
    mx.eval(QtQ, I)
    return float(mx.max(mx.abs(QtQ - I)).item())


class TestOrthogonalityBench(unittest.TestCase):
    def test_bench(self):
        shapes = [(512, 128), (1024, 256)]
        for (m, n) in shapes:
            X = mx.random.normal(shape=(m, n)).astype(mx.float32)
            print(f"\n[Ortho Bench] shape=({m},{n})")
            t_cols = _median_time(lambda: orthonormal_columns(X))
            Q = orthonormal_columns(X)
            err_cols = _orth_err_cols(Q)
            print(f"  orthonormal_columns: {t_cols:.4f}s, err={err_cols:.2e}")

            t_blk = _median_time(lambda: orthogonalize_blocked(X, B=32))
            Qb = orthogonalize_blocked(X, B=32)
            err_blk = _orth_err_cols(Qb)
            print(f"  orthogonalize_blocked: {t_blk:.4f}s, err={err_blk:.2e}")

            t_init = _median_time(lambda: orthogonal((m, n)))
            W = orthogonal((m, n))
            err_init = _orth_err_cols(W)
            print(f"  orthogonal initializer: {t_init:.4f}s, err={err_init:.2e}")


if __name__ == "__main__":
    unittest.main()

