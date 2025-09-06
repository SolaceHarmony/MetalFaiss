"""
test_kernel_benchmarks.py - Micro-benchmarks for MLX vs Metal-kernel paths

Benchmarks QR (and optionally SVD) when both MLX-only and Metal-kernel
implementations are present. Uses perf_counter and forces MLX evaluation.

Notes:
- Enable external Metal kernel QR via env: METALFAISS_USE_GPU_QR=1
- The test will skip the kernel path if not available or if import fails.
"""

import os
import time
import unittest
import mlx.core as mx

from ..faissmlx.qr import pure_mlx_qr


def _time_it(fn, *args, repeats=3, warmup=1, **kwargs):
    # Warmup
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        # Force evaluation of MLX lazy graph
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        # Force eval to include compute time
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    # Return median to reduce noise
    return times[len(times) // 2]


class TestKernelBenchmarks(unittest.TestCase):
    def setUp(self):
        # Use a moderately large but quick matrix to exercise GPU tiling
        self.m, self.n = 1024, 256
        self.A = mx.random.normal(shape=(self.m, self.n)).astype(mx.float32)

    def test_qr_benchmark(self):
        # MLX QR (pure Modified Gram–Schmidt)
        t_mlx = _time_it(pure_mlx_qr, self.A)

        # Verify correctness roughly: Q^T Q ≈ I
        Q_mlx, R_mlx = pure_mlx_qr(self.A)
        mx.eval(Q_mlx, R_mlx)
        qtq = mx.matmul(mx.transpose(Q_mlx[:, :min(self.m, self.n)]), Q_mlx[:, :min(self.m, self.n)])
        self.assertTrue(bool(mx.allclose(qtq, mx.eye(qtq.shape[0]), rtol=1e-3, atol=1e-3)))

        print(f"\n[QR Benchmark] MLX: {t_mlx:.4f}s")


if __name__ == "__main__":
    unittest.main()
