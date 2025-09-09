# Common Pitfalls (and Fixes)

Collected sharp edges when working with MLX in practice.

## 1) Silent Broadcasting Mismatch

Symptom: an op “works” but results are wrong.

Fix: print shapes, add `keepdims`, or reshape explicitly.

```python
assert x.shape == y.shape or x.shape == (1, y.shape[-1])
```

## 2) Non‑Contiguous Views Hurt Performance

Symptom: indexes/slices feed into heavy ops with unexpected slowness.

Fix: `mx.contiguous(view)` before compute‑heavy kernels.

## 3) Loss Isn’t Scalar

Symptom: gradient transform fails or grads have surprising shapes.

Fix: ensure your loss reduces to a scalar (mean/sum over batch/time).

## 4) RNG Reproducibility

Symptom: data‑dependent tests or demos aren’t reproducible.

Fix: use and reuse explicit random keys.

```python
key = mx.random.key(0)
x1 = mx.random.normal((32, 32), key=key)
```

Also avoid calling `mx.random.seed` inside functions transformed by `mx.vmap`, `mx.value_and_grad`, or `mx.compile`. Seeding mutates global state and can collapse independence or break caching. Prefer passing `key=` (and splitting) as arguments.

## 5) Device Surprises

Symptom: results differ across runs or machines.

Fix: print `mx.default_device()`, scope with `with mx.default_device(...)` during experiments.

## 6) Parameter Tree Mismatch

Symptom: optimizer update errors due to tree shape differences.

Fix: keep params as the single source of truth; pass them to `module.apply` and only update via the optimizer’s return value.

## 7) Mixing MLX and NumPy Mutates Behind the Graph

Symptom: gradients seem wrong; JIT/lazy optimizations don’t kick in.

Cause: turning an MLX array into a zero‑copy NumPy view and mutating it changes buffers outside MLX’s graph.

Fix:
- Keep compute in MLX inside differentiable functions.
- If you must cross boundaries, copy: `np.array(x, copy=True)` and `mx.array(np_arr.copy())` on the way back.
- Copy MLX arrays with `mx.array(x)` or `mx.copy(x)`; there is no `.clone()`.

## 8) Python Loops for Conditional Elementwise Ops

Symptom: very slow code when applying `if/else` per element.

Fix: use `mx.where` — it’s vectorized, lazy (fuseable), and avoids branch divergence on GPU by selecting between precomputed branches.

```python
# Bad: Python loop
out = [xi if xi > 0 else 0.1 * xi for xi in data]

# Good: vectorized
import mlx.core as mx
x = mx.array(data)
y = mx.where(x > 0, x, 0.1 * x)
```

See `mlx.core.where` for performance notes.
