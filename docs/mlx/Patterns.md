# MLX Patterns — Quick Cheat Sheet

Fast reference for replacing Python/NumPy idioms with MLX patterns.

## Arithmetic / Math
- `a + b` → `mx.add(a, b)`
- `a - b` → `mx.subtract(a, b)`
- `a * b` → `mx.multiply(a, b)`
- `a / b` → `mx.divide(a, b)`
- `a @ b` / `np.dot(a,b)` → `mx.matmul(a, b)`

## Comparisons / Masks
- `a == b` → `mx.equal(a, b)`
- `a != b` → `mx.not_equal(a, b)`
- `a < b` → `mx.less(a, b)`
- `a <= b` → `mx.less_equal(a, b)`
- `a > b` → `mx.greater(a, b)`
- `a >= b` → `mx.greater_equal(a, b)`
- Combine: `mx.logical_and(m1, m2)`, `mx.logical_or(m1, m2)`
- Select: `mx.where(mask, x, y)`

## Casting / Scalars
- On device: `x.astype(mx.float32)` / `x.astype(mx.int32)`
- Python boundary: `float(x.astype(mx.float32))`, `int(x.astype(mx.int32))`
- Build scalars as MLX arrays: `mx.array(0, dtype=...)`, `inf = mx.ones(() )/ mx.zeros(())`

## Modulo / Floor‑Divide (arrays)
- `%` or `//` → `q, r = mx.divmod(a, b)`

## Booleans
- `bool(mask)` → `mx.any(mask)` or `mx.all(mask)`

## Bitwise (uint types)
- `x & y` → `mx.bitwise_and(x, y)`
- `x | y` → `mx.bitwise_or(x, y)`
- `x ^ y` → `mx.bitwise_xor(x, y)`
- `~x` → `mx.bitwise_not(x)`

## Top‑K values + indices
```python
order = mx.argsort(D, axis=axis)[..., :k]
vals = mx.take_along_axis(D, order, axis=axis)
```
Or use project helper: `values, indices = mlx_topk(D, k, axis=1)`.

## Slicing
- `mx.slice(a, start_indices, axes, slice_size)`
  - `start_indices`: MLX array; `axes`, `slice_size`: Python lists
- `mx.slice_update(a, update, start_indices, axes)`

## Reductions
- `mx.sum`, `mx.max`, `mx.min`, `mx.mean`, `mx.var`, `mx.std` (with axis/keepdims)

## Control‑flow inside compile/vmap
- Avoid `if mx.sum(mask) == 0:` — replace with masked updates and `mx.where`
- Preallocate buffers; use masks to “no‑op” invalid entries

## Devices and Streams (performance)
- Prefer the default Metal device for heavy ops; avoid `stream=mx.cpu` in hot paths.
- Use `mlx.core.metal.is_available()` and `mlx.core.metal.device_info()` to introspect.
- Keep compute on the same device; avoid device hopping in tight loops.
