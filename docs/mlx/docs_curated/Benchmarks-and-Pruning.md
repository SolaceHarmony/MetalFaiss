Benchmarks & Pruning: Measure, Then Keep the Winner

Methodology

- Warm up (build graph/JIT) before timing.
- Use median of N runs; force `mx.eval` to include compute time.
- Benchmark the shapes that matter to you (m, n, k grid), not generic sizes.
- Record results (CSV/Markdown); update autoswitch thresholds accordingly.

Example harness (unittest-style)

```python
def _time(fn, *args, repeats=3, warmup=1, **kwargs):
    for _ in range(warmup):
        out = fn(*args, **kwargs); mx.eval(out) if not isinstance(out, tuple) else mx.eval(*out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(*args, **kwargs)
        mx.eval(out) if not isinstance(out, tuple) else mx.eval(*out)
        times.append(time.perf_counter() - t0)
    times.sort(); return times[len(times)//2]
```

Policy

- If Metal kernel wins clearly at your target sizes, keep it and gate MLX by threshold.
- If MLX wins, keep it and drop the kernel path (or leave behind a guarded flag).
- Revisit after device updates (e.g., new GPU drivers) — performance trade-offs can flip.

In this repo

- `python/metalfaiss/unittest/test_svd_benchmarks.py`
- `python/metalfaiss/unittest/test_kernel_benchmarks.py`
- `docs/research/Journal.md` logs iterations, mistakes, and decisions.

