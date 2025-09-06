"""
Benchmark MLX subspace SVD vs Metal kernel Z-step.

Runs a small timing to compare the two paths and prints results.
"""

import os
import time
import unittest
import mlx.core as mx

from ..faissmlx.svd import topk_svd


def _time_svd(A, k, iters, use_kernel, repeats=3, warmup=1):
    # Warmup
    for _ in range(warmup):
        U, S, Vt = topk_svd(A, k, iters=iters, use_kernel=use_kernel)
        mx.eval(U, S, Vt)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        U, S, Vt = topk_svd(A, k, iters=iters, use_kernel=use_kernel)
        mx.eval(U, S, Vt)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times)//2]


class TestSVDBenchmarks(unittest.TestCase):
    def test_svd_benchmark(self):
        shapes = [
            (256, 128, 16),
            (512, 256, 32),
        ]
        for (m, n, k) in shapes:
            A = mx.random.normal(shape=(m, n)).astype(mx.float32)

            t_mlx = _time_svd(A, k, iters=3, use_kernel=False)
            t_kernel = _time_svd(A, k, iters=3, use_kernel=True)
            # Banded kernel timing (choose band size heuristically)
            band = 8 if k <= 16 else 16
            t_kernel_banded = _time_svd(A, k, iters=3, use_kernel=True)

            print(f"\n[SVD Benchmark] shape=({m}x{n},k={k}) MLX: {t_mlx:.4f}s, Kernel-tiled: {t_kernel:.4f}s")
            # Re-run with banded kernel via env var (serial)
            import os as _os
            _os.environ['METALFAISS_SVD_BAND'] = str(band)
            t_kernel_banded = _time_svd(A, k, iters=3, use_kernel=True)
            print(f"[SVD Benchmark] banded (band={band}): {t_kernel_banded:.4f}s")
            # Re-run with banded + streams (S=4) if available
            _os.environ['METALFAISS_SVD_STREAMS'] = '4'
            t_kernel_streams = _time_svd(A, k, iters=3, use_kernel=True)
            print(f"[SVD Benchmark] banded+streams (band={band}, S=4): {t_kernel_streams:.4f}s")
            del _os.environ['METALFAISS_SVD_BAND']
            del _os.environ['METALFAISS_SVD_STREAMS']

            # Sanity: both should produce shapes
            U0, S0, Vt0 = topk_svd(A, k, iters=2, use_kernel=False)
            U1, S1, Vt1 = topk_svd(A, k, iters=2, use_kernel=True)
            self.assertEqual(U0.shape, (m, k))
            self.assertEqual(S0.shape, (k,))
            self.assertEqual(Vt0.shape, (k, n))
            self.assertEqual(U1.shape, (m, k))
            self.assertEqual(S1.shape, (k,))
            self.assertEqual(Vt1.shape, (k, n))


if __name__ == "__main__":
    unittest.main()
