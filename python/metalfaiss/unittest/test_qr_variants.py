"""
test_qr_variants.py — Correctness and timing for QR variants

Runs MLX two-pass MGS vs kernel-assisted projection+update for a couple shapes.
We assert orthonormality and reconstruction quality; we print timings for research.
"""

import os
import time
import unittest
import mlx.core as mx

from ..faissmlx.qr import pure_mlx_qr


def _qr_reconstruction_error(A, Q, R):
    A_hat = mx.matmul(Q, R)
    num = mx.sqrt(mx.sum((A_hat - A) * (A_hat - A)))
    den = mx.sqrt(mx.sum(A * A)) + 1e-6
    return num / den


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter(); out = fn(*args, **kwargs); mx.eval(*out); t1 = time.perf_counter()
    return out, (t1 - t0)


class TestQRVariants(unittest.TestCase):
    def _run_case(self, m, n):
        A = mx.random.normal(shape=(m, n)).astype(mx.float32)
        # Baseline MLX two-pass MGS
        (Q0, R0), t0 = _time(pure_mlx_qr, A)
        k0 = min(m, n)
        Qk0 = Q0[:, :k0]
        # Orthonormal check (left orthonormal)
        qtq0 = mx.matmul(mx.transpose(Qk0), Qk0)
        self.assertTrue(bool(mx.allclose(qtq0, mx.eye(k0), rtol=1e-3, atol=1e-3)))
        rerr0 = _qr_reconstruction_error(A, Q0, R0)

        # Kernel-assisted path
        os.environ['METALFAISS_USE_QR_KERNEL'] = '1'
        (Q1, R1), t1 = _time(pure_mlx_qr, A)
        del os.environ['METALFAISS_USE_QR_KERNEL']
        k1 = min(m, n)
        Qk1 = Q1[:, :k1]
        qtq1 = mx.matmul(mx.transpose(Qk1), Qk1)
        self.assertTrue(bool(mx.allclose(qtq1, mx.eye(k1), rtol=1e-3, atol=1e-3)))
        rerr1 = _qr_reconstruction_error(A, Q1, R1)

        print(f"\n[QR Variants] shape=({m}x{n}) MLX: {t0:.4f}s (recon={float(rerr0):.3e}) | "
              f"Kernel: {t1:.4f}s (recon={float(rerr1):.3e})")

        # Reconstruction within reasonable tolerance
        self.assertTrue(bool(mx.all(mx.less_equal(rerr0, mx.array(1e-2, dtype=mx.float32))).item()))  # boundary-ok
        self.assertTrue(bool(mx.all(mx.less_equal(rerr1, mx.array(1e-2, dtype=mx.float32))).item()))  # boundary-ok

    def test_qr_tall_skinny(self):
        self._run_case(512, 64)

    def test_qr_square(self):
        self._run_case(256, 256)


if __name__ == "__main__":
    unittest.main()
