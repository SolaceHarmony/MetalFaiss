# Training Patterns

Practical, composable patterns for training with MLX: functional autograd, batching, evaluation, and logging.

## Functional Autograd

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW

net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
params = net.parameters()
opt = AdamW(1e-3)

def loss_fn(p, x, y):
    logits = net.apply(p, x)
    return mx.mean((logits - y) ** 2)

for step in range(1000):
    x = mx.random.normal((128, 32))
    y = mx.random.normal((128, 10))
    loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
    params = opt.update(params, grads)
    if step % 100 == 0:
        print(step, loss.item())
```

## Evaluation Mode

Set modules to eval to disable dropout/update‑style behavior in norms.

```python
net.eval()
val_loss = loss_fn(params, val_x, val_y)
```

## Accumulation (Large Effective Batch)

Accumulate grads over micro‑batches and update once.

```python
def grad_fn(p, x, y):
    return mx.value_and_grad(loss_fn)(p, x, y)[1]

acc = None
for i, (x, y) in enumerate(loader):
    g = grad_fn(params, x, y)
    acc = g if acc is None else mx.utils.tree_map(mx.add, acc, g)
    if (i + 1) % k == 0:
        acc = mx.utils.tree_map(lambda g: g / k, acc)
        params = opt.update(params, acc)
        acc = None
```

## Checkpointing

```python
state = {"params": params, "opt": opt.state()}
mx.save("ckpt.npz", state)

restored = mx.load("ckpt.npz")
params = restored["params"]
# Recreate optimizer, then restore its state if you saved it
opt = AdamW(1e-3)
# opt.load_state(restored["opt"])  # if supported in your version
```

## Logging and Metrics

Convert scalars via `.item()` and accumulate on host for logging.

```python
train_loss += loss.item()
```

## Precision

Use `float16`/`bfloat16` arrays for memory compute benefits; ensure numerically sensitive reductions remain in `float32` where required.

## Randomness in Training Loops (keys + split)

- Prefer explicit keys over global seeding to guarantee independence and reproducibility across vectorized/parallel code.
- Split once per step and thread the new key forward.

```python
import mlx.core as mx

def step(params, batch, key):
    x, y = batch
    k_aug, k_drop, k_next = mx.random.split(key, num=3)  # independent streams

    # Example: per‑sample augmentation with vmap + subkeys
    aug = mx.vmap(lambda k_i, x_i: x_i + 0.1 * mx.random.normal(x_i.shape, key=k_i))
    sub = mx.random.split(k_aug, num=x.shape[0])
    x_aug = aug(sub, x)

    def loss_fn(p):
        logits = net.apply(p, x_aug)
        # dropout keyed explicitly if needed, e.g., mx.nn.dropout(logits, key=k_drop)
        return mx.mean((logits - y) ** 2)

    value, grads = mx.value_and_grad(loss_fn)(params)
    return value, grads, k_next

# JIT compile the step if available in your version
compiled_step = mx.compile(step) if hasattr(mx, "compile") else step

key = mx.random.key(0)
for it, batch in enumerate(loader):
    loss, grads, key = compiled_step(params, batch, key)
    params = opt.update(params, grads)
```

Notes:
- Use `mx.random.split` to create independent substreams for each stochastic component (augmentations, dropout, noise).
- For per‑example randomness, feed a batch of subkeys to `mx.vmap`.

## Purity and Interop Boundaries

- Keep differentiable compute pure‑MLX inside `loss_fn`/`step` to preserve lazy transforms and JIT opportunities.
- Avoid NumPy/Python mutations of MLX buffers inside `value_and_grad`, `vmap`, or compiled functions.

Pitfall (breaks gradients/optimizations):
```python
def bad_loss(p, x):
    x_view = np.array(x, copy=False)   # shares MLX buffer
    x_view[:] = 0.0                    # out‑of‑graph mutation
    return mx.sum(net.apply(p, x))
```

Safe boundary patterns:
- To NumPy: `x_host = np.array(x, copy=True)` when you need host ops.
- Back to MLX: `x_mx = mx.array(x_host.copy())` to decouple and re‑enter the graph.
- Copy MLX arrays with `mx.array(x)` or `mx.copy(x)` (no `.clone()`).

## Device Routing and Precision

- No `device=` keyword: set default device or use scoped context; some ops accept `stream=` (e.g., CPU routing for double‑precision).
```python
mx.set_default_device(mx.gpu)           # if available
with mx.default_device(mx.cpu):         # CPU‑scope for float64
    loss64 = mx.sum(mx.ones((8,), dtype=mx.float64))
```
