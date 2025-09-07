"""
bench_compile.py — Compare compiled vs non‑compiled MLX functions we use.

Runs small, repeatable benchmarks for:
- GELU elementwise
- SVD iteration (MLX path), and kernel wrapper
- FlatIndex search wrapper (toy)

Writes CSV/PNG to docs/benchmarks/compile_benefits.* and prints a summary.
"""

from __future__ import annotations
import time
import os
from typing import Tuple, List
import mlx.core as mx


def _median(f, warmup: int = 5, reps: int = 30):
    for _ in range(warmup):
        out = f()
        mx.eval(out) if isinstance(out, mx.array) else None
    ts: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        # Trigger computation
        if isinstance(out, tuple):
            for o in out:
                if isinstance(o, mx.array):
                    mx.eval(o)
        elif isinstance(out, mx.array):
            mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


def bench_gelu() -> Tuple[float, float]:
    def gelu(x: mx.array) -> mx.array:
        return mx.divide(
            mx.multiply(x, mx.add(mx.array(1.0), mx.erf(mx.divide(x, mx.sqrt(mx.array(2.0)))))),
            mx.array(2.0)
        )

    x = mx.random.uniform(shape=(32, 1000, 4096)).astype(mx.float32)
    t0 = _median(lambda: gelu(x))
    cgelu = mx.compile(gelu)
    t1 = _median(lambda: cgelu(x))
    return t0 * 1e3, t1 * 1e3


def bench_svd_step() -> Tuple[float, float, float, float]:
    from metalfaiss.faissmlx.svd import topk_svd
    A = mx.random.uniform(shape=(512, 256)).astype(mx.float32)
    # MLX path
    t_mlx = _median(lambda: topk_svd(A, k=32, iters=3, use_kernel=False, use_compile=False))
    t_mlx_c = _median(lambda: topk_svd(A, k=32, iters=3, use_kernel=False, use_compile=True))
    # Kernel path wrapper
    t_ker = _median(lambda: topk_svd(A, k=32, iters=3, use_kernel=True, use_compile=False))
    t_ker_c = _median(lambda: topk_svd(A, k=32, iters=3, use_kernel=True, use_compile=True))
    return t_mlx * 1e3, t_mlx_c * 1e3, t_ker * 1e3, t_ker_c * 1e3


def bench_flat_wrapper() -> Tuple[float, float]:
    # Toy example: compile a wrapper over index.search for repeated calls
    from metalfaiss.indexflat import FlatIndex
    idx = FlatIndex(d=64)
    xb = mx.random.uniform(shape=(20000, 64)).astype(mx.float32)
    idx.add(xb)

    def search_fun(xq: mx.array):
        # Pure-MLX search path
        res = idx.search(xq, k=10)
        # Support both legacy and MLX-native result types
        d = getattr(res, 'distances', None)
        i = getattr(res, 'indices', None)
        if i is None:
            i = getattr(res, 'labels', None)
        return d, i

    xq = mx.random.uniform(shape=(200, 64)).astype(mx.float32)
    t0 = _median(lambda: search_fun(xq))
    cfun = mx.compile(search_fun)
    t1 = _median(lambda: cfun(xq))
    return t0 * 1e3, t1 * 1e3


def main():
    out_dir = "docs/benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Tuple[str, float, float]] = []

    # GELU
    g0, g1 = bench_gelu()
    rows.append(("gelu", g0, g1))

    # SVD
    s0, s1, k0, k1 = bench_svd_step()
    rows.append(("svd_mlx", s0, s1))
    rows.append(("svd_kernel", k0, k1))

    # Flat wrapper
    f0, f1 = bench_flat_wrapper()
    rows.append(("flat_wrapper", f0, f1))

    # Print
    print("\nCompile benefit (ms per call):")
    for name, t_nc, t_c in rows:
        sp = (t_nc / t_c) if t_c > 0 else 0.0
        print(f"- {name:12s}: non-compiled={t_nc:.3f}  compiled={t_c:.3f}  speedup={sp:.2f}x")

    # Write CSV
    csv_path = os.path.join(out_dir, "compile_benefits.csv")
    with open(csv_path, "w") as f:
        f.write("name,non_compiled_ms,compiled_ms,speedup\n")
        for name, t_nc, t_c in rows:
            sp = (t_nc / t_c) if t_c > 0 else 0.0
            f.write(f"{name},{t_nc:.6f},{t_c:.6f},{sp:.3f}\n")

    # Plot PNG if matplotlib exists
    try:
        import matplotlib.pyplot as plt
        names = [r[0] for r in rows]
        t_nc = [r[1] for r in rows]
        t_c = [r[2] for r in rows]
        x = range(len(names))
        plt.figure(figsize=(7,4))
        w = 0.35
        plt.bar([i-w/2 for i in x], t_nc, width=w, label='non-compiled')
        plt.bar([i+w/2 for i in x], t_c, width=w, label='compiled')
        plt.xticks(list(x), names, rotation=15, ha='right')
        plt.ylabel('ms per call (median)')
        plt.title('Compile benefit')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'compile_benefits.png'), dpi=160)
        plt.close()
    except Exception:
        pass

    print(f"\nWrote {csv_path} and optional PNG to docs/benchmarks")


if __name__ == "__main__":
    main()
