"""
Autotune benchmark for GEMM tiling and banding/streams options.

Prints timing tables and best tile selections per shape.
"""

import os
import time
import unittest
import itertools
import mlx.core as mx

from ..faissmlx.kernels import gemm_kernels as gk


def _median_time(fn, warmup=1, repeats=3):
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


def _bench_zstep(m, n, k, av_tile, atb_tile):
    # Set tiles then rebuild kernels
    gk.set_gemm_tiles(av=av_tile, atb=atb_tile)
    # Allocate inputs once
    A = mx.random.normal(shape=(m, n)).astype(mx.float32)
    V = mx.random.normal(shape=(n, k)).astype(mx.float32)

    def run():
        B = gk.gemm_av(A, V)
        Z = gk.gemm_at_b(A, B)
        return Z

    t = _median_time(run)
    return t


class TestKernelAutotuneBench(unittest.TestCase):
    def test_autotune_tiles(self):
        shapes = [
            (256, 128, 16),
            (512, 256, 32),
        ]
        av_tiles = ["16x16", "32x8", "8x32"]
        atb_tiles = ["16x16", "8x32", "32x8"]

        for (m, n, k) in shapes:
            print(f"\n[Autotune] shape=({m}x{n},k={k})")
            best = (None, None, 1e9)
            for av in av_tiles:
                for atb in atb_tiles:
                    t = _bench_zstep(m, n, k, av, atb)
                    print(f"  AV={av:>5} ATB={atb:>5}  Z-step: {t:.4f}s")
                    if t < best[2]:
                        best = (av, atb, t)
            print(f"  -> best tiles: AV={best[0]}, ATB={best[1]}  time={best[2]:.4f}s")


if __name__ == "__main__":
    unittest.main()

