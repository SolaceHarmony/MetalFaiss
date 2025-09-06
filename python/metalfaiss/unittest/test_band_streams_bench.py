"""
Benchmark band size and streams for SVD kernel path.

Prints timing table across (band, streams).
"""

import os
import time
import unittest
import mlx.core as mx

from ..faissmlx.svd import topk_svd


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


class TestBandStreamsBench(unittest.TestCase):
    def test_band_streams(self):
        shapes = [
            (256, 128, 16),
            (512, 256, 32),
        ]
        bands = [None, 8, 16, 32]
        streams = [None, 2, 4]

        for (m, n, k) in shapes:
            print(f"\n[Band/Streams] shape=({m}x{n},k={k})")
            A = mx.random.normal(shape=(m, n)).astype(mx.float32)
            # Baseline MLX
            t_mlx = _median_time(lambda: topk_svd(A, k, iters=3, use_kernel=False))
            print(f"  MLX baseline: {t_mlx:.4f}s")

            for b in bands:
                for s in streams:
                    if b is None and s is not None:
                        continue  # streams only meaningful with banding
                    # Set env overrides for the kernel path
                    if b is not None:
                        os.environ['METALFAISS_SVD_BAND'] = str(b)
                    else:
                        os.environ.pop('METALFAISS_SVD_BAND', None)
                    if s is not None:
                        os.environ['METALFAISS_SVD_STREAMS'] = str(s)
                    else:
                        os.environ.pop('METALFAISS_SVD_STREAMS', None)

                    t = _median_time(lambda: topk_svd(A, k, iters=3, use_kernel=True))
                    tag = f"band={b if b is not None else 'mono'}"
                    if s is not None:
                        tag += f", streams={s}"
                    print(f"  {tag:<18}: {t:.4f}s")

            # Cleanup env
            os.environ.pop('METALFAISS_SVD_BAND', None)
            os.environ.pop('METALFAISS_SVD_STREAMS', None)


if __name__ == "__main__":
    unittest.main()

