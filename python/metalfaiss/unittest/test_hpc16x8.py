"""
test_hpc16x8.py — Validate HPC16x8 helpers (kahan_sum, safe_norm2)

We construct numerically tricky sequences to show compensated summation
reduces error vs naive mx.sum. These tests run on GPU with MLX scalars.
"""

import unittest
import math
import mlx.core as mx

from ..faissmlx.hpc16x8 import kahan_sum, safe_norm2


class TestHPC16x8(unittest.TestCase):
    def test_kahan_vs_naive(self):
        # Sequence with catastrophic cancellation
        # Sum of pairs (1e8, -1e8) interleaved with small ones should be ~N_small
        Npairs = 1000
        Nsmall = 1000
        big = mx.array([1e8, -1e8] * Npairs, dtype=mx.float32)
        small = mx.ones((Nsmall,), dtype=mx.float32)
        x = mx.concatenate([big, small], axis=0)

        # Reference in Python double
        ref = (0.0) + Nsmall * 1.0
        # Naive MLX sum
        naive = mx.sum(x)
        # Kahan MLX sum
        kh = kahan_sum(x)

        naive_err = abs(float(naive) - ref)
        kahan_err = abs(float(kh) - ref)
        self.assertLessEqual(kahan_err, naive_err)

    def test_safe_norm2(self):
        v = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
        n2 = safe_norm2(v)
        self.assertEqual(float(n2), 0.0)
        n2e = safe_norm2(v, eps=1e-6)
        # Allow small rounding; ensure it's close to eps and > 0
        self.assertGreater(float(n2e), 0.0)
        self.assertLess(abs(float(n2e) - 1e-6), 2e-7)


if __name__ == "__main__":
    unittest.main()
