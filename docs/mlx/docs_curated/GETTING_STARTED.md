# Getting Started with MLX

This guide gets you productive fast: install MLX, run your first array ops on Apple GPU, and understand just enough about devices and performance to avoid common pitfalls.

## Prerequisites

- Apple Silicon (M‑series) recommended; MLX targets Metal on macOS.
- Recent macOS (Ventura or newer recommended).
- Python 3.9+ (typical ML setups use 3.10/3.11).

## Install

```bash
pip install mlx mlx-nn
```

Optional extras (examples, extras):

```bash
# Jupyter / notebooks
pip install jupyterlab ipykernel

# Plotting
pip install matplotlib
```

Verify install:

```python
import mlx.core as mx
a = mx.array([1, 2, 3])
print(a, a.dtype, a.shape)
```

## First Steps: Arrays and Math

```python
import mlx.core as mx

x = mx.arange(0, 6).reshape(2, 3)
y = mx.ones_like(x)
z = x + 2 * y
print(z)

# Reductions
print(mx.sum(z))        # scalar
print(mx.mean(z, axis=0))

# Broadcasting works like NumPy
v = mx.array([0., 1., 2.])
print(z + v)
```

Tip: MLX arrays are immutable views; many ops return new arrays without copying data when possible.

## Devices and Streams (Quick)

MLX targets Metal automatically when available.

```python
import mlx.core as mx

print(mx.default_device())     # e.g., Device(gpu:0) on Apple Silicon
```

To run on CPU explicitly:

```python
with mx.default_device(mx.cpu):
    out = mx.ones((1024, 1024)) @ mx.ones((1024, 1024))
```

Streams enable overlapping compute and host work. Start simple; introduce streams if you profile contention.

```python
s = mx.new_stream()
with s:
    big = mx.ones((4096, 4096)) @ mx.ones((4096, 4096))
mx.synchronize()  # wait if you need results on host
```

## Random, Dtypes, Shapes

```python
key = mx.random.key(0)
noise = mx.random.normal((2, 3), key=key)
print(noise.dtype, noise.shape)

# Dtype control
x = mx.ones((3, 3), dtype=mx.float16)
y = mx.asarray(x, dtype=mx.float32)

# Reshape/view ops
z = mx.reshape(y, (9,))          # view when possible
w = mx.transpose(x, (1, 0))
```

## Performance Notes

- Prefer larger batch/array operations over Python loops.
- Reuse arrays and keys to reduce allocations.
- Use `mx.clear_cache()` sparingly; it forces cache purges.
- Profile hotspots; if a function exists in `mx.fast.*`, try it.

## Minimal Training Loop Example

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam

class MLP(nn.Module):
    def __init__(self, d_in=32, d_h=64, d_out=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_out)
        )
    def __call__(self, x):
        return self.layers(x)

model = MLP()
opt = Adam(1e-3)

def loss_fn(params, x, y):
    logits = model.apply(params, x)
    # simple MSE for demo
    return mx.mean((logits - y) ** 2)

params = model.parameters()

for step in range(200):
    x = mx.random.normal((128, 32))
    y = mx.random.normal((128, 10))
    l, grads = mx.value_and_grad(loss_fn)(params, x, y)
    params = opt.update(params, grads)
    if step % 50 == 0:
        print(step, l.item())
```

Next: see Arrays for a deeper tour of the core API, or jump to Neural Networks for training patterns.

## The Apple Way (TL;DR)

- No `device=`: choose device globally (`mx.set_default_device`), with a scoped context (`with mx.default_device(...)`), or per‑op `stream` when supported.
- Immutability: arrays aren’t mutated and there is no `.clone()`; copying is `mx.array(x)` or `mx.copy(x)`.
- Lazy by default: ops record a graph; compute happens at eval points (`mx.eval`, `.item()`). Keep differentiable code pure‑MLX for best transforms/JIT.
- RNG: Prefer explicit `key=` and `mx.random.split` in parallel code, or wrap a small `Generator` shim if you want Torch‑style call sites.
- Dtype/device: float64 is CPU‑only; route double‑precision work to CPU via default device or `stream=mx.cpu`.
- Transforms: use `mx.value_and_grad`, `mx.vmap`, and `mx.compile` (when available) to build fast, functional training steps.
- Modules: `module.apply(params, x)` is pure; optimizers return new `params` trees.
- Indexing helpers: `mx.slice`/`mx.slice_update` take MLX arrays for indices and Python lists for axes/sizes.
- Metal: for GPU‑heavy custom ops, prefer stable scalar inner loops + parallel block updates; add guardrails and dbg buffers.

Quick links:
- Arrays: copying, views vs contiguous, interop boundaries (NumPy)
- Random: key/split, compatibility RNG shim, factories (`rand`, `randn`, `*_like`)
- Linalg: QR/Cholesky on Metal, HPC16x8 pattern for stability
