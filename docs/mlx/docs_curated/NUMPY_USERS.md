# MLX for NumPy Users: A Practical Guide

## Introduction
MLX is Apple’s machine learning framework designed for Apple Silicon. It mirrors much of NumPy’s API and semantics but has important differences in function signatures, parameter types, devices, and randomness that can trip up NumPy habits. This guide highlights practical mappings and the key gotchas.

## Array Creation

| NumPy | MLX | Notes |
|-------|-----|------|
| `np.array([1, 2, 3])` | `mx.array([1, 2, 3])` | Similar |
| `np.zeros((3, 3))` | `mx.zeros((3, 3))` | Similar |
| `np.ones((3, 3))` | `mx.ones((3, 3))` | Similar |
| `np.eye(3)` | `mx.eye(3)` | Similar |
| `np.arange(10)` | `mx.arange(10)` | Similar |
| `np.linspace(0, 1, 10)` | `mx.linspace(0, 1, 10)` | Similar |

## Indexing and Slicing

| NumPy | MLX | Notes |
|-------|-----|------|
| `a[0, 1]` | `a[0, 1]` | Basic indexing is similar |
| `a[1:3, 2:4]` | `a[1:3, 2:4]` | Basic slicing is similar |
| `np.take(a, idx, axis=0)` | `mx.take(a, idx, axis=0)` | Similar |
| — | `mx.slice(a, start_indices, axes, slice_size)` | MLX‑specific helper (see example) |
| — | `mx.slice_update(a, update, start_indices, axes)` | MLX‑specific helper (see example) |

Notes for `mx.slice`/`mx.slice_update`:
- `start_indices`: MLX array (e.g., `mx.array([row, col, ...])`).
- `axes`: Python list/tuple of axes to slice along.
- `slice_size`: Python list/tuple with lengths along those axes.

MLX slicing differences vs Python/NumPy:
- MLX arrays are immutable; `a[1:3] = ...` does not mutate. Use `a.at[1:3].set(update)` or `mx.slice_update`.
- Boolean masked assignment: use `mx.where` to form a new array conditionally.

Examples:
```python
import mlx.core as mx

a = mx.array([1,2,3,4,5])
b = a.at[1:4].set(mx.array([9,9,9]))   # array([1,9,9,9,5])
mask = a > 3
c = mx.where(mask, 0, a)               # zero out >3
```

### Example: `mx.slice` and `mx.slice_update`
```python
import mlx.core as mx

# Create a test array
a = mx.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Extract a slice starting at (row=0, col=1) with size (2, 2)
start_indices = mx.array([0, 1])
axes = [0, 1]          # Python list
slice_size = [2, 2]    # Python list
b = mx.slice(a, start_indices, axes, slice_size)
# b == [[2, 3], [5, 6]]

# Update a slice in place (functionally: returns a new array)
update = mx.array([[10, 11],
                   [12, 13]])
c = mx.slice_update(a, update, start_indices, axes)
# c == [[ 1, 10, 11],
#       [ 4, 12, 13],
#       [ 7,  8,  9]]
```

## Mathematical Operations

Both operator syntax and function calls work in MLX. Use whichever reads best for you.

| NumPy | MLX (operators) | MLX (functions) |
|-------|------------------|-----------------|
| `a + b` | `a + b` | `mx.add(a, b)` |
| `a - b` | `a - b` | `mx.subtract(a, b)` |
| `a * b` | `a * b` | `mx.multiply(a, b)` |
| `a / b` | `a / b` | `mx.divide(a, b)` |
| `np.matmul(a, b)` | `a @ b` | `mx.matmul(a, b)` |
| `np.exp(a)` | `mx.exp(a)` | — |
| `np.log(a)` | `mx.log(a)` | — |
| `np.sin(a)` | `mx.sin(a)` | — |

## Reductions

| NumPy | MLX | Notes |
|-------|-----|------|
| `np.sum(a, axis=0)` | `mx.sum(a, axis=0)` | Similar |
| `np.max(a, axis=0)` | `mx.max(a, axis=0)` | Similar |
| `np.min(a, axis=0)` | `mx.min(a, axis=0)` | Similar |
| `np.mean(a, axis=0)` | `mx.mean(a, axis=0)` | Similar |
| `np.var(a, axis=0)` | `mx.var(a, axis=0)` | Similar |
| `np.std(a, axis=0)` | `mx.std(a, axis=0)` | Similar |

## Shape Manipulation

| NumPy | MLX | Notes |
|-------|-----|------|
| `np.reshape(a, (2, 3))` | `mx.reshape(a, (2, 3))` | Similar |
| `np.transpose(a)` | `mx.transpose(a)` | Similar |
| `np.concatenate([a, b], axis=0)` | `mx.concatenate([a, b], axis=0)` | Similar |
| `np.stack([a, b], axis=0)` | `mx.stack([a, b], axis=0)` | Similar |
| `np.split(a, 3, axis=0)` | `mx.split(a, 3, axis=0)` | Similar |

## Key Differences and Gotchas

1) Parameter and argument types
- Some MLX helpers (e.g., `mx.slice`, `mx.slice_update`) require specific types: MLX arrays for indices; Python lists/tuples for axes and sizes.

2) Python operators vs MLX functions
- MLX supports operators (`+`, `-`, `*`, `/`, `@`) and function forms (`mx.add`, …). Pick consistently; both are fine.

3) MLX arrays vs Python scalars
- To get a host scalar, prefer `.item()` on 0‑D arrays instead of `int(...)`/`float(...)` on an MLX array.
- Example: `loss_value = loss.item()`.

4) Devices
- No `device=` parameter on most ops. Control placement via default device, scoped device context, or per‑op `stream` when supported.
- MLX targets Metal on Apple Silicon by default. You can scope placement:
```python
with mx.default_device(mx.cpu):
    x = mx.ones((2, 3))
mx.set_default_device(mx.cpu)  # global
# or per‑op, when supported
y = mx.sum(x, stream=mx.cpu)
```

5) Randomness
- MLX provides both a global seed (`mx.random.seed(...)`) and explicit keys (`mx.random.key(seed)` passed into ops). Prefer explicit keys for reproducibility in composed code.

6) Views and contiguity
- Slicing returns strided views. For kernels that require/benefit from contiguous input, use `mx.contiguous(view)`.

7) Framework boundaries and copies
- Crossing MLX⇄NumPy breaks lazy graph assumptions if you share memory. Avoid zero‑copy views when mutating.
- Safe patterns:
  - To NumPy: `x_host = np.array(x, copy=True)` to prevent back‑mutation.
  - From NumPy: `x = mx.array(np_arr.copy())` to decouple lifetimes.
  - To copy an MLX array: `mx.array(x)` or `mx.copy(x)`; there is no `.clone()`.

Example (pitfall vs safe):
```python
import numpy as np, mlx.core as mx
x = mx.array([3.0])

def impure(x):
    np.array(x, copy=False)[:] = 0.0  # external mutation, not tracked
    return mx.sum(x)

val, grad = mx.value_and_grad(impure)(x)  # val=0.0 but grad≈1.0 (wrong)

def safe(x):
    x_copy = mx.array(x)  # copy before any NumPy interop
    x_host = np.array(x_copy, copy=True)
    x_host[:] = 0.0
    return mx.sum(x)      # original x unchanged; grad correct
```

## Testing MLX Code

Try minimal examples when learning ops:
```python
import mlx.core as mx

# Test mx.slice
a = mx.array([[1, 2, 3], [4, 5, 6]])
start = mx.array([0, 1])
axes = [0, 1]  # Python list
size = [1, 2]  # Python list
result = mx.slice(a, start, axes, size)
print(result)  # [[2, 3]]

# Test mx.slice_update
update = mx.array([[10, 11]])
result = mx.slice_update(a, update, start, axes)
print(result)  # [[1, 10, 11], [4, 5, 6]]
```

For more, see: Arrays, Shapes & Broadcasting, Devices & Streams, and Porting from PyTorch in this curated set.
