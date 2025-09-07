# MLX for NumPy Users: A Practical Guide

This guide summarizes key differences between NumPy and MLX, and shows the “right” MLX patterns to avoid host conversions and make your code compile- and vmap‑friendly. It’s safe to copy into PRs and link in reviews.

## Array Creation

| NumPy | MLX | Notes |
|---|---|---|
| `np.array([1,2,3])` | `mx.array([1,2,3])` | Same idea |
| `np.zeros((3,3))` | `mx.zeros((3,3))` | |
| `np.ones((3,3))` | `mx.ones((3,3))` | |
| `np.eye(3)` | `mx.eye(3)` | |
| `np.arange(10)` | `mx.arange(10)` | |
| `np.linspace(0,1,10)` | `mx.linspace(0,1,10)` | |

## Indexing and Slicing

| NumPy | MLX | Key Differences |
|---|---|---|
| `a[0,1]` | `a[0,1]` | Same for basic indexing |
| `a[1:3,2:4]` | `a[1:3,2:4]` | Same for slices |
| `np.take(a, idx, axis=0)` | `mx.take(a, idx, axis=0)` | Same semantics |
| — | `mx.slice(a, start_indices, axes, slice_size)` | MLX‑specific: `start_indices` is an MLX array; `axes` and `slice_size` are Python lists/tuples |
| — | `mx.slice_update(a, upd, start_indices, axes)` | MLX‑specific; same param rules |

Example:

```python
start = mx.array([0, 1])      # MLX array
axes = [0, 1]                 # Python list
size = [2, 2]                 # Python list
b = mx.slice(a, start, axes, size)
c = mx.slice_update(a, update, start, axes)
```

## Math Operations (Prefer MLX functions over Python operators)

| NumPy/Python | MLX |
|---|---|
| `a + b` | `mx.add(a, b)` |
| `a - b` | `mx.subtract(a, b)` |
| `a * b` | `mx.multiply(a, b)` |
| `a / b` | `mx.divide(a, b)` |
| `a @ b`, `np.dot(a,b)` | `mx.matmul(a, b)` |
| `np.exp(a)` | `mx.exp(a)` |
| `np.log(a)` | `mx.log(a)` |

Reductions: `mx.sum`, `mx.max`, `mx.min`, `mx.mean`, `mx.var`, `mx.std` mirror NumPy.

## Comparisons and Logic (Very Important)

Use MLX comparisons for masks:
- Equal / Not equal: `mx.equal(a, b)`, `mx.not_equal(a, b)`
- Order: `mx.less`, `mx.less_equal`, `mx.greater`, `mx.greater_equal`
- Combine masks: `mx.logical_and`, `mx.logical_or`, `mx.logical_not`
- Select by mask: `mx.where(cond, x, y)`

Do not use `==`, `!=`, `<`, `<=`, `>`, `>=` directly on MLX arrays inside compute code. The linter will flag these and suggest the correct `mx.*` alternative.

## Casting and Scalars

- On device (preferred): `x.astype(mx.float32)`, `x.astype(mx.int32)`
- At Python boundary (only when necessary): `float(x.astype(mx.float32))`, `int(x.astype(mx.int32))`
- Build constants as MLX scalars:
  - `zero = mx.array(0, dtype=mx.float32)`
  - `inf = mx.divide(mx.ones((), dtype=mx.float32), mx.zeros((), dtype=mx.float32))`

Never call `float(x)` or `int(x)` on an MLX array in compute paths — do it only at API boundaries.

## Modulo and Floor‑Divide

- Arrays: `q, r = mx.divmod(a, b)` (use `r` as remainder, `q` as floor‑quotient)
- Avoid `%` and `//` on MLX arrays.

## Top‑K Values and Indices

- `mx.topk` returns values only. To get indices:
  ```python
  order = mx.argsort(D, axis=1)[:, :k]
  vals = mx.take_along_axis(D, order, axis=1)
  ```
- Use our helper `mlx_topk(D, k, axis=1)` which returns `(values, indices)` and handles row‑wise gather efficiently.

## Booleans

- Reduce masks: `mx.any(mask)`, `mx.all(mask)`
- Never use `bool(mask)` on an MLX array.

## Device/Host Boundaries

- Keep everything as MLX arrays. Convert to Python types only at the edges (printing, API return that requires Python lists, etc.).
- When compiling or vmapping, avoid Python control flow on array values — use masked operations and `mx.where`.

## Quick Recipes

```python
# Equality mask
mask = mx.equal(labels, mx.array(3, dtype=labels.dtype))
sel = mx.where(mask, a, b)

# L2 distances: ||X||^2 + ||Y||^2 - 2 X Y^T
nx = mx.sum(mx.square(X), axis=1, keepdims=True)
ny = mx.sum(mx.square(Y), axis=1, keepdims=True)
D = mx.add(nx, mx.transpose(ny))
D = mx.subtract(D, mx.add(mx.matmul(X, Y.T), mx.matmul(X, Y.T)))

# Top‑k (values + indices)
order = mx.argsort(D, axis=1)[:, :k]
vals = mx.take_along_axis(D, order, axis=1)

# Remainder / quotient
q, r = mx.divmod(a, b)
```

## Testing

Test small shapes interactively; prefer masked ops and MLX functions to keep code compile/vmap‑friendly.

