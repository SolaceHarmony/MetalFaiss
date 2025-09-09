# Shapes, Layout, and Broadcasting

Explicit shape thinking keeps MLX code correct and easy to port. This guide emphasizes concrete shapes and broadcasting behavior.

## Shape Basics

- Use `a.shape`, `a.ndim`, and `a.size` liberally while iterating.
- Many ops follow NumPy semantics; reductions accept `axis` and often `keepdims`.

```python
x = mx.arange(0, 12).reshape(3, 4)
print(x.shape)           # (3, 4)
print(mx.mean(x, 0).shape)  # (4,)
print(mx.mean(x, 0, keepdims=True).shape)  # (1, 4)
```

## Broadcasting Rules (NumPy‑like)

Trailing axes align; size‑1 axes expand to match.

```python
m = mx.ones((2, 3))
v = mx.array([10, 20, 30])
out = m + v           # (2, 3)
```

If shapes don’t align, add explicit axes:

```python
v2 = mx.reshape(v, (1, 3))
out = m + v2
```

## Views vs Contiguity

Slicing creates views with strides. Some kernels prefer contiguous inputs.

```python
view = mx.arange(0, 12).reshape(3, 4)[:, ::2]  # (3, 2) strided
contig = mx.contiguous(view)                    # compact layout
```

## Convolutions and Batches

When building CNNs, verify your tensor order and layer expectations. A common convention is `(N, C, H, W)` for inputs.

```python
N, C, H, W = 8, 3, 32, 32
imgs = mx.random.normal((N, C, H, W))
y = nn.Conv2d(C, 32, kernel_size=3, padding=1)(imgs)
```

Always print intermediate shapes during bring‑up.

## Masking and Indexing

```python
mask = (mx.arange(10) % 2 == 0)
evens = mx.where(mask, mx.arange(10), 0)
topk_vals, topk_idx = mx.topk(mx.random.normal((128,)), k=5)
```

## Reductions and Axes

```python
a = mx.random.normal((2, 3, 4))
print(mx.sum(a, axis=(1, 2)))       # (2,)
print(mx.max(a, axis=-1).shape)     # (2, 3)
```

## Gradient Shapes

`mx.value_and_grad(fn)` returns gradients matching the shape tree of parameters. If you get mismatches, check that your loss reduces to a scalar and that batch dimensions are consistent.

## For NumPy Users

- Broadcasting is NumPy‑like. If shapes won’t align, insert singleton dims with `mx.reshape` or `mx.expand_dims`.
- Reductions: pass `axis` as int/tuple; `keepdims=True` behaves like NumPy.
- `mx.stack`, `mx.concatenate`, `mx.split` mirror NumPy signatures.
