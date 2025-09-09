# In‑Place Ops and Mutability in MLX

MLX favors a functional style: ops return new arrays, parameter updates return new trees. This reduces hidden side effects and makes code easier to reason about.

## No In‑Place Tensor Ops

- PyTorch idioms like `x.add_()` or `param.data -= ...` do not apply.
- Use pure expressions and rebind:

```python
x = x + y * 0.1
params = opt.update(params, grads)
```

## No `.clone()`; how to copy

- MLX arrays are immutable; there is no `.clone()`.
- To copy, construct a new array: `y = mx.array(x)` or use `y = mx.copy(x)`.

```python
import mlx.core as mx
x = mx.array([1, 2, 3])
y = mx.array(x)  # copy
z = mx.copy(x)   # same effect
```

## Views vs Copies

- Indexing/slicing returns views with strides. Most elementwise ops can consume views; some kernels prefer contiguous inputs.

```python
view = x[:, ::2]
contig = mx.contiguous(view)
```

## Updates: Slices and Masks

- Slice replacement returns a new array:

```python
x = mx.array([1, 2, 3, 4])
y = x.at[1:3].set(mx.array([-1, -2]))   # x unchanged; y is new
```

- Explicit helper form mirrors the API: `mx.slice_update(a, update, start_indices, axes)`.

- Boolean mask updates use `mx.where`:

```python
mask = x > 2
z = mx.where(mask, 0, x)                # conditional new array
```

## Framework Boundaries (NumPy/Python)

- Zero‑copy NumPy views can mutate MLX buffers outside the graph, breaking gradients/optimizations:

```python
import numpy as np, mlx.core as mx
x = mx.array([3.0])
def bad_fn(x):
    np.array(x, copy=False)[:] = 0.0  # external mutation (not tracked)
    return mx.sum(x)
val, grad = mx.value_and_grad(bad_fn)(x)  # val=0 but grad≈1 (wrong)
```

Fixes:
- Keep math in MLX; avoid mutating via NumPy.
- If you must cross the boundary, copy explicitly: `x_host = np.array(x, copy=True)`; when returning, decouple: `x_mx = mx.array(x_host.copy())`.
- Inside differentiable fns, never perform out‑of‑graph mutations.

## Module State

- `module.parameters()` is a snapshot; you choose when to replace it after an optimizer update.
- `module.apply(params, ...)` runs with an explicit parameter set; no hidden mutation.

## Benefits

- Safer parallelism and fewer heisenbugs from hidden state.
- Easier checkpoint diffs and reproducibility.
