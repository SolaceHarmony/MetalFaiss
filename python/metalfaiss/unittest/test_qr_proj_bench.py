"""
Benchmark QR projection kernels: simple vs simdgroup reduction.

Prints timings and checks basic correctness against MLX dot.
"""

import time
import unittest
import os
import mlx.core as mx

from ..faissmlx.kernels.qr_kernels import project_coeffs


def _median_time(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


class TestQRProjBench(unittest.TestCase):
    def test_proj_variants(self):
        shapes = [
            (1024, 64),
            (4096, 128),
        ]
        for (m, k) in shapes:
            print(f"\n[QR Proj] m={m}, k={k}")
            Q = mx.random.normal(shape=(m, k)).astype(mx.float32)
            v = mx.random.normal(shape=(m,)).astype(mx.float32)

            # Reference MLX
            t_mlx = _median_time(lambda: mx.matmul(mx.transpose(Q), v))
            print(f"  MLX dot: {t_mlx:.4f}s")

            # Simple kernel
            os.environ['METALFAISS_QR_DOT'] = 'simple'
            t_simple = _median_time(lambda: project_coeffs(Q, v))
            print(f"  Kernel simple: {t_simple:.4f}s")

            # SIMD kernel
            os.environ['METALFAISS_QR_DOT'] = 'simd'
            t_simd = _median_time(lambda: project_coeffs(Q, v))
            print(f"  Kernel simd:   {t_simd:.4f}s")

            # Correctness check
            ref = mx.matmul(mx.transpose(Q), v)
            ks = project_coeffs(Q, v)  # simd mode still set
            self.assertTrue(bool(mx.allclose(ref, ks, rtol=1e-3, atol=1e-3)))

        os.environ.pop('METALFAISS_QR_DOT', None)


if __name__ == "__main__":
    unittest.main()

