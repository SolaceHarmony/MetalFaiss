"""
test_svd_variants.py — Correctness checks across SVD variants

Goal: incubate the banded/streamed kernels by verifying that multiple
implementations produce acceptable approximations. We avoid asserting
performance to keep the test stable, but we print timings to aid research.
"""

import os
import time
import unittest
import mlx.core as mx

from ..faissmlx.svd import topk_svd


def _residual_ratio(A, U, S, Vt):
    V = mx.transpose(Vt)
    AV = mx.matmul(A, V)
    US = U * S.reshape((1, -1))
    num = mx.sqrt(mx.sum((AV - US) * (AV - US)))
    den = mx.sqrt(mx.sum(A * A)) + 1e-6
    return num / den


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter(); out = fn(*args, **kwargs); mx.eval(*out); t1 = time.perf_counter()
    return out, (t1 - t0)


class TestSVDVariants(unittest.TestCase):
    def _run_case(self, m, n, k, band=8, streams=4):
        A = mx.random.normal(shape=(m, n)).astype(mx.float32)

        # MLX baseline (no compile)
        (U0, S0, Vt0), t0 = _time(topk_svd, A, k, iters=4, use_kernel=False)
        r0 = _residual_ratio(A, U0, S0, Vt0)

        # MLX + compile (if available)
        (U0c, S0c, Vt0c), t0c = _time(topk_svd, A, k, iters=4, use_kernel=False, use_compile=True)
        r0c = _residual_ratio(A, U0c, S0c, Vt0c)

        # Kernel paths
        (U1, S1, Vt1), t1 = _time(topk_svd, A, k, iters=4, use_kernel=True)
        r1 = _residual_ratio(A, U1, S1, Vt1)

        os.environ['METALFAISS_SVD_BAND'] = str(band)
        (U2, S2, Vt2), t2 = _time(topk_svd, A, k, iters=4, use_kernel=True)
        r2 = _residual_ratio(A, U2, S2, Vt2)
        del os.environ['METALFAISS_SVD_BAND']

        os.environ['METALFAISS_SVD_BAND'] = str(band)
        os.environ['METALFAISS_SVD_STREAMS'] = str(streams)
        (U3, S3, Vt3), t3 = _time(topk_svd, A, k, iters=4, use_kernel=True)
        r3 = _residual_ratio(A, U3, S3, Vt3)
        del os.environ['METALFAISS_SVD_BAND']
        del os.environ['METALFAISS_SVD_STREAMS']

        print(
            f"\n[SVD Variants] shape=({m}x{n},k={k}) "
            f"MLX: {t0:.4f}s (r={float(r0):.3f}); MLX+cmp: {t0c:.4f}s (r={float(r0c):.3f}); "
            f"K-mono: {t1:.4f}s (r={float(r1):.3f}); K-band: {t2:.4f}s (r={float(r2):.3f}); "
            f"K-band+S{streams}: {t3:.4f}s (r={float(r3):.3f})"
        )

        for r in [r0, r0c, r1, r2, r3]:
            self.assertTrue(bool(mx.all(mx.less_equal(r, mx.array(0.6, dtype=mx.float32))).item()))  # boundary-ok

    def test_small(self):
        self._run_case(256, 128, 16, band=8, streams=4)

    def test_medium(self):
        self._run_case(1024, 256, 32, band=16, streams=4)

    def test_large(self):
        # Keep iterations modest for runtime
        self._run_case(2048, 512, 64, band=32, streams=4)


if __name__ == "__main__":
    unittest.main()
