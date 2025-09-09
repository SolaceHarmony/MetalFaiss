# Arrays in MLX

MLX arrays are the foundation: typed, shaped, device‑aware containers with NumPy‑like semantics and fast Metal kernels.

## Creation

```python
import mlx.core as mx

mx.array([1, 2, 3])                # from Python data
mx.zeros((2, 3))                    # zeros
mx.ones((2, 3), dtype=mx.float16)   # ones with dtype
mx.full((2, 3), 7)                  # constant fill
mx.arange(0, 10, 2)                 # [0,2,4,6,8]
mx.linspace(0, 1, 5)                # 5 points from 0..1
mx.eye(4)                           # identity
```

Conversions:

```python
mx.asarray(numpy_array)             # wrap/copy as needed
mx.array(mx_existing_array)         # ensures MLX type
```

## Dtype and Shape

```python
x = mx.ones((3, 3), dtype=mx.float32)
print(x.dtype, x.shape, x.ndim, x.size, x.nbytes)
y = x.astype(mx.float16)
```

Common dtypes: `mx.float32`, `mx.float16`, `mx.bfloat16`, `mx.int32`, `mx.int64`, `mx.bool_`.

## Indexing and Slicing

```python
a = mx.arange(0, 12).reshape(3, 4)
print(a[0, 1:3])      # view
b = a[:, ::2]         # stride view
c = mx.take(a, mx.array([0, 3, 5]))
```

Views are cheap; use `mx.contiguous(a)` if you need compact memory for kernels that require it.

## Reshape, Transpose, Move Axes

```python
x = mx.arange(0, 6).reshape(2, 3)
y = mx.transpose(x, (1, 0))
z = mx.moveaxis(y, 0, -1)
w = mx.reshape(z, (3, 2, 1))
```

## Broadcasting

Broadcasting follows NumPy rules (trailing axes aligned, size‑1 axes expand).

```python
v = mx.array([10, 20, 30])
m = mx.ones((2, 3))
print(m + v)  # shape (2, 3)
```

## Math and Elementwise Ops

```python
x, y = mx.arange(0, 5), mx.arange(1, 6)
print(x + y, x - y, x * y, x / y)
print(mx.exp(x), mx.log1p(y), mx.sqrt(y))
print(mx.sin(x), mx.tanh(y))
print(mx.where(x % 2 == 0, x, -x))
```

Comparison and logic:

```python
mx.equal(x, y), mx.less(x, y), mx.greater_equal(x, y)
mx.logical_and(x > 1, y < 4)
```

## Reductions

```python
a = mx.arange(0, 24).reshape(2, 3, 4)
print(mx.sum(a), mx.mean(a), mx.std(a), mx.var(a))
print(mx.max(a, axis=1), mx.min(a, axis=2))
print(mx.argmax(a, axis=-1), mx.argmin(a))
```

`keepdims` is supported in many reductions; consult the function signature.

## Random

```python
key = mx.random.key(42)
u = mx.random.uniform((2, 3), low=-1.0, high=1.0, key=key)
n = mx.random.normal((2, 3), key=key)
cat = mx.random.categorical(mx.array([0.2, 0.5, 0.3]), (5,), key=key)
```

Use and reuse `key` to get deterministic sequences.

## Linear Algebra (Pointers)

```python
A = mx.random.normal((4, 4))
b = mx.random.normal((4,))
x = mx.linalg.solve(A, b)
U, S, Vh = mx.linalg.svd(A)
```

See the Linear Algebra guide for more.

## FFT (Pointers)

```python
signal = mx.random.normal((1024,))
spec = mx.fft.fft(signal)
```

See the FFT guide for multidimensional transforms and tips.

## Contiguity and Memory

Slicing often produces strided views. Some kernels expect contiguous inputs for peak performance.

```python
view = mx.arange(0, 12).reshape(3, 4)[:, ::2]
contig = mx.contiguous(view)
```

`mx.clear_cache()` purges internal caches; it’s rarely needed during normal runs.

## Copying and Immutability

- Arrays are immutable; there is no `.clone()`.
- Copy via constructor or `mx.copy`:

```python
import mlx.core as mx
x = mx.array([1, 2, 3])
y1 = mx.array(x)
y2 = mx.copy(x)
```

Interop rule of thumb:
- Avoid zero‑copy NumPy views if you will mutate; use `np.array(x, copy=True)` instead.
- When constructing from NumPy, decouple with `mx.array(np_arr.copy())` to avoid shared buffers.

## Exporting and Interop

```python
mx.save("arr.npz", {"a": mx.arange(10)})
loaded = mx.load("arr.npz")
```

Use `save_safetensors` for tensor‑only checkpoints, or `save_gguf` for model formats compatible with GGML tooling.

## For NumPy Users

- Operators vs functions: both work (`x + y` or `mx.add(x, y)`; `x @ y` or `mx.matmul(x, y)`).
- Indexing/slicing is familiar; MLX also exposes `mx.slice` and `mx.slice_update` with specific arg types:
  - `start_indices`: MLX array (e.g., `mx.array([row, col])`)
  - `axes`/`slice_size`: Python list/tuple
- Reductions mirror NumPy: `mx.sum`, `mx.mean`, `mx.std`, etc.; use `keepdims` when needed.
- Extract host scalars with `.item()` on 0‑D arrays instead of `int(...)`/`float(...)` on MLX arrays.

## Random Factories (Convenience)

MLX uses functional RNG ops (e.g., `mx.random.normal(shape=...)`). If you miss NumPy/Torch factories, add small helpers:

```python
def rand(shape, low=0.0, high=1.0, dtype=mx.float32, key=None):
    return mx.random.uniform(shape=shape, low=low, high=high, key=key).astype(dtype)
def randn(shape, mean=0.0, std=1.0, dtype=mx.float32, key=None):
    return mx.random.normal(shape=shape, loc=mean, scale=std, key=key).astype(dtype)
def rand_like(x, low=0.0, high=1.0, key=None):
    return mx.random.uniform(shape=x.shape, low=low, high=high, key=key).astype(x.dtype)
```

These keep dtype/device consistent with a reference array and slot into NumPy/Torch‑ported code.
