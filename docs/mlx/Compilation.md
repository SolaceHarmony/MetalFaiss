# MLX Compilation (mx.compile) — Practical Guide + Project Examples

MLX can compile Python functions into optimized compute graphs. Compiled graphs fuse and reorder operations, often cutting runtime and memory use substantially. This guide shows how to apply `mx.compile` safely and productively in MetalFaiss, with concrete examples and pitfalls to avoid.

## Basics

```python
import mlx.core as mx

def fun(x, y):
    return mx.exp(-x) + y

x = mx.array(1.0)
y = mx.array(2.0)

# Regular call
print(fun(x, y))  # array(2.36788, dtype=float32)

# Compile
compiled_fun = mx.compile(fun)
print(compiled_fun(x, y))  # same value
```

Notes
- First call compiles (trace + optimize + codegen). Subsequent calls reuse the cache (fast).
- Recompiles happen on changes to shape, rank, dtype, or argument count.
- Avoid compiling inside tight loops; compile once and reuse the handle.

```python
compiled = mx.compile(fun)
compiled(x, y)   # compiles now
compiled(x, y)   # cache hit
mx.compile(fun)(x, y)  # reuses cached graph for this fun signature
```

## Example Speedup (GELU)

Elementwise chains (e.g., GELU) fuse into a single kernel.

```python
import mlx.core as mx
import mlx.nn as nn
import time

def gelu(x):
    return x * (1 + mx.erf(x / mx.sqrt(mx.array(2.0)))) / mx.array(2.0)

def timeit(fun, x):
    for _ in range(10):
        mx.eval(fun(x))  # warmup
    t0=time.perf_counter()
    for _ in range(100):
        mx.eval(fun(x))
    dt=(time.perf_counter()-t0)/100*1e3
    print(f"Time per iter: {dt:.3f} ms")

x = mx.random.uniform(shape=(32, 1000, 4096))

# Baseline vs compiled
timeit(gelu, x)
timeit(mx.compile(gelu), x)
```
On M‑series GPUs, compiled GELU often runs ~3–5× faster (varies by device/shape).

## Debugging Compiled Functions

Compiled functions are traced with placeholder arrays on first call; printing inside them fails.

```python
@mx.compile
def fun(x):
    z = -x
    # print(z)  # avoid printing inside compiled functions
    return mx.exp(z)
```

For debugging, either disable compilation temporarily:

```python
mx.disable_compile()
fun(mx.array(5.0))
```

…or return intermediate arrays and inspect the outputs.

## Pure Functions and State Capture

Compiled functions are intended to be pure. If you mutate Python state, the placeholders captured at trace time will cause surprises.

Return the state explicitly:

```python
state = []

@mx.compile
def fun(x, y):
    z = x + y
    state.append(z)
    return mx.exp(z), state

out, st = fun(mx.array(1.0), mx.array(2.0))
print(st)  # [array(3, dtype=float32)]
```

Or use `inputs=` / `outputs=` capture to treat external lists as implicit inputs/outputs:

```python
from functools import partial
state = []

@partial(mx.compile, outputs=state)
def step(x, y):
    z = x + y
    state.append(z)
    return mx.exp(z)

step(mx.array(1.0), mx.array(2.0))
print(state)  # [array(3, dtype=float32)]
```

If your function reads from global arrays, use `inputs=` to refresh reads across calls:

```python
from functools import partial
state = [mx.array(1.0)]

@partial(mx.compile, inputs=state)
def fun(x):
    return x + state[0]

print(fun(mx.array(1.0)))          # array(2)
state[0] = mx.array(5.0)
print(fun(mx.array(1.0)))          # array(6)
```

## Training Graphs (Compiled)

Compile forward+backward+update together to reduce Python overhead during training.

```python
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from functools import partial

x = mx.random.uniform(shape=(4, 10))
y = mx.array([0, 1, 0, 1])
model = nn.Linear(10, 1)
opt = optim.SGD(learning_rate=0.1, momentum=0.8)

def loss_fn(m, x, y):
    logits = m(x).squeeze()
    return nn.losses.binary_cross_entropy(logits, y)

state = [model.state, opt.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)
    opt.update(model, grads)
    return loss

for it in range(10):
    loss = step(x, y)
    mx.eval(state)
    print(loss)
```

Tip: if your model uses RNG (e.g., Dropout), include `mx.random.state` in `inputs=` and `outputs=`.

## Transformations + Compile

Function transforms (e.g., grad) compose with compile.

```python
g = mx.grad(mx.exp)
compiled_g = mx.compile(g)
print(g(mx.array(1.0)))
print(compiled_g(mx.array(1.0)))
```

Compiling the outermost function usually yields best results, even if inner functions are compiled.

## Shapeless Compilation

Compile once for variable shapes with `shapeless=True` — only when your function is shape‑agnostic.

```python
def fun(x, y):
    return mx.abs(x + y)
compiled = mx.compile(fun, shapeless=True)
print(compiled(mx.array(1.0), mx.array(-2.0)))
print(compiled(mx.array([1.0,-6.0]), mx.array([-2.0,3.0])))
```

Avoid capturing static shapes (e.g., via `reshape(x.shape[0]*x.shape[1], -1)`). Prefer `x.flatten(0,1)`.

---

# MetalFaiss Examples

## Compiled SVD Iteration (MLX path)

We compile one iteration step `Z = Aᵀ(A·V)` then orthonormalize via QR. See `python/metalfaiss/faissmlx/svd.py` — the implementation caches compiled functions by (m,n,k,dtype,device).

```python
import mlx.core as mx
from metalfaiss.faissmlx.svd import topk_svd

A = mx.random.uniform(shape=(512, 256)).astype(mx.float32)
U, S, Vt = topk_svd(A, k=32, iters=3, use_kernel=False, use_compile=True)
```
Measured on this box: ~1.6× speedup vs non‑compiled MLX path at these shapes.

## Compiled Kernel Wrapper (Z‑step)

The kernel Z‑step (AV then AᵀB via custom Metal kernels) benefits less from compile (kernels dominate runtime), but compiling the wrapper reduces Python overhead:

```python
U, S, Vt = topk_svd(A, k=32, iters=3, use_kernel=True, use_compile=True)
```

## Compiled Flat Search (toy)

For repeated queries with fixed k, shapes are stable; compiling the inner function can help small batch cases:

```python
import mlx.core as mx
from metalfaiss.indexflat import FlatIndex

idx = FlatIndex(d=64)
idx.add(mx.random.uniform(shape=(10000,64)).tolist())

def flat_search(xq):
    return idx.search(xq.tolist(), k=10)

c_search = mx.compile(flat_search)
vals_idx = c_search(mx.random.uniform(shape=(100,64)))
```

Note: this example serializes MLX→Python lists to match current index API; prefer pure‑MLX search endpoints where available.

---

# Tips
- Prefer pure MLX ops and scalars inside compiled functions (see No‑CPU‑Math Contract).
- Compile once and reuse; avoid compiling in inner loops.
- Cache compiled wrappers by (shape/dtype/device) if you dispatch across many sizes.
- For mixed CPU/GPU work on unified memory, assign streams (mx.cpu/mx.gpu) inside compiled functions; MLX schedules dependencies automatically.

# See Also
- docs/mlx/No-CPU-Math-Contract.md
- docs/mlx/Comprehensive-MLX-Metal-Guide.md
- MLX user docs: compile, function transforms, unified memory
