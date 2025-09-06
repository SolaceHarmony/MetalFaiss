"""
Autotune benchmark for GEMM tiling and banding/streams options.

Prints timing tables and best square T + flag selections per shape.
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


def _bench_zstep(m, n, k, sqT, db=False, v4=False, pad_atb=False):
    # Set square tile + flags then rebuild kernels
    os.environ["METALFAISS_USE_GEMM_KERNEL"] = "1"
    os.environ["METALFAISS_GEMM_TILE_SQ"] = str(sqT)
    os.environ["METALFAISS_GEMM_DB"] = "1" if db else "0"
    os.environ["METALFAISS_GEMM_V4"] = "1" if v4 else "0"
    os.environ["METALFAISS_GEMM_PAD_ATB"] = "1" if pad_atb else "0"
    gk.reset_gemm_kernels()
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
        Ts = [8, 16, 32]
        toggles = [
            (False, False, False),
            (True,  False, False),
            (False, True,  False),
            (False, False, True ),
            (True,  True,  False),
            (True,  False, True ),
        ]

        for (m, n, k) in shapes:
            print(f"\n[Autotune] shape=({m}x{n},k={k})")
            best = (None, 1e9)
            for T in Ts:
                for (db, v4, pad) in toggles:
                    t = _bench_zstep(m, n, k, T, db=db, v4=v4, pad_atb=pad)
                    print(f"  T={T:>2} DB={int(db)} V4={int(v4)} PAD={int(pad)}  Z: {t:.4f}s")
                    if t < best[1]:
                        best = (f"T={T} db={int(db)} v4={int(v4)} pad={int(pad)}", t)
            print(f"  -> best: {best[0]}  time={best[1]:.4f}s")


if __name__ == "__main__":
    unittest.main()
