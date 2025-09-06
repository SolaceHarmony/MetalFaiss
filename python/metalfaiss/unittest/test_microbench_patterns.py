"""
Microbenchmarks for teaching patterns (non-assertive; prints timings)

Focus: integer division/modulus vs 2D grid mapping for index remap.

These are informative and not strict performance tests. They print median
timings to help humans see the effect locally.
"""

import time
import unittest
import mlx.core as mx


_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_SRC_DIV = r"""
  uint gid = thread_position_in_grid.x;
  uint n = shape[0];
  uint k = shape[1];
  uint total = n * k;
  if (gid >= total) return;
  uint col = gid % k;     // non-constant modulus
  uint row = gid / k;     // non-constant division
  out[gid] = in0[row * k + col] * 2.0f;
"""

_SRC_2D = r"""
  uint2 g = thread_position_in_grid.xy;
  uint n = shape[0];
  uint k = shape[1];
  if (g.x >= k || g.y >= n) return;
  uint idx = g.y * k + g.x;
  out[idx] = in0[idx] * 2.0f;
"""


def _median_time(fn, reps=5, warmup=1):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


class TestMicrobenchPatterns(unittest.TestCase):
    def test_int_div_vs_2d(self):
        # Size large enough to see a difference quickly
        n, k = 2048, 2048
        arr = mx.random.normal((n*k,)).astype(mx.float32)
        shape = mx.array([n, k], dtype=mx.uint32)

        ker_div = mx.fast.metal_kernel(
            name="divmap", input_names=["in0","shape"], output_names=["out"],
            header=_HEADER, source=_SRC_DIV, ensure_row_contiguous=True)

        ker_2d = mx.fast.metal_kernel(
            name="twod", input_names=["in0","shape"], output_names=["out"],
            header=_HEADER, source=_SRC_2D, ensure_row_contiguous=True)

        # Warmup
        (y,) = ker_div(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype],
                       grid=((n*k+255)//256*256,1,1), threadgroup=(256,1,1)); mx.eval(y)
        (y,) = ker_2d(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype],
                       grid=((k+31)//32*32, (n+31)//32*32,1), threadgroup=(32,32,1)); mx.eval(y)

        t_div = _median_time(lambda: ker_div(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype],
                                              grid=((n*k+255)//256*256,1,1), threadgroup=(256,1,1)))
        t_2d  = _median_time(lambda: ker_2d(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype],
                                              grid=((k+31)//32*32, (n+31)//32*32,1), threadgroup=(32,32,1)))

        print(f"\n[Microbench] int-div map: {t_div:.4f}s; 2D map: {t_2d:.4f}s")


if __name__ == "__main__":
    unittest.main()

