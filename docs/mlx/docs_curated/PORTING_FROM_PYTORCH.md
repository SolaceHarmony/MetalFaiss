# Porting from PyTorch to MLX

This guide maps common PyTorch patterns to MLX and calls out differences that often trip people up when moving between the two ecosystems.

## Top Mental Model Shifts (PyTorch → MLX)

- Device control: no `device=` on ops; set default device, use scoped contexts, or per‑op `stream`.
- Mutability: no in‑place tensor ops or `.clone()`; expressions return new arrays; copy with `mx.array(x)`/`mx.copy(x)`.
- RNG: use explicit `key=` (or a small `Generator` wrapper) instead of global `manual_seed`.
- Lazy eval: graphs build until `mx.eval`/`.item()`; keep differentiable code pure‑MLX.
- Float64: CPU‑only; dispatch double‑precision work to CPU.
- Modules: `module.apply(params, x)`; optimizers return updated parameter trees.
- Indexing helpers: `mx.slice`/`mx.slice_update` types differ from NumPy (`start_indices` as MLX array; axes/size as Python lists).
- Transforms: `mx.value_and_grad`, `mx.vmap`, `mx.compile` cover autograd, batching, and JIT.

## Cheat Sheet (PyTorch → MLX)

- `loss.backward()` → `value, grads = mx.value_and_grad(loss_fn)(params, batch)`
- `optimizer.step()` → `params = opt.update(params, grads)`
- `optimizer.zero_grad()` → not needed; grads are returned, not stored on tensors
- `tensor.to(device)` / `.cuda()` / `.cpu()` → set default or scoped device: `mx.set_default_device(...)` / `with mx.default_device(...)`
- `torch.manual_seed(n)` → `key = mx.random.key(n)`; split per use: `k1, k2 = mx.random.split(key, num=2)`
- `torch.Generator` → small wrapper `Generator(seed)` that hands out keys internally (see Random doc)
- `x[mask] = v` → `x2 = mx.where(mask, v, x)` (returns new array)
- `x[idx] = v` (slice) → `x2 = x.at[idx].set(v)` or `mx.slice_update(x, v, start_indices, axes)`
- `keepdim=` / `dim=` → `keepdims=` (when supported) / `axis=` in MLX
- `state_dict()`/`load_state_dict()` → `module.save_weights()/load_weights()` or `mx.save/mx.load` for trees
- `.item()` logging → same: `loss.item()`; also triggers eval
- Views + in‑place ops → MLX is functional; avoid in‑place patterns and rely on pure expressions


## Mental Model

- PyTorch: stateful tensors track gradients when `requires_grad=True`; optimizers update parameters in place.
- MLX: arrays are regular values; use functional transforms (e.g., `mx.value_and_grad`) to compute gradients, and optimizers return updated parameter trees.

## Autograd: Imperative vs Functional

PyTorch:

```python
loss = model(x).log_softmax(-1).gather(1, y[:,None]).mean()
loss.backward()
optimizer.step(); optimizer.zero_grad()
```

MLX:

```python
def loss_fn(params, x, y):
    logits = model.apply(params, x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
params = opt.update(params, grads)
```

Key difference: gradients aren’t “attached” to arrays; you compute them with a transform and pass them to the optimizer.

## Parameters and State

PyTorch: `state_dict()` / `load_state_dict()` mutate module state; optimizers are mutable.

MLX:

- `module.parameters()` returns a pure parameter tree.
- `module.state()` includes parameters plus buffers.
- `module.apply(params, ...)` runs a module with explicit params.
- Optimizers return new params (and maintain internal state internally).

## Devices

PyTorch often defaults to CPU; you explicitly move to CUDA/MPS.

MLX selects Metal GPU on Apple Silicon when available. You can scope device selection:

```python
with mx.default_device(mx.cpu):
    y = model.apply(params, x)

No `.to(device)` needed:
- MLX uses unified memory and dispatches to the appropriate backend. Prefer setting a default device or scoped context rather than moving arrays around.
- Float64 is CPU‑only; scope double‑precision work to CPU via `with mx.default_device(mx.cpu)` or `stream=mx.cpu` on supported ops.
```

## RNG and Reproducibility

PyTorch uses global RNGs with `torch.manual_seed`.

MLX favors explicit keys:

```python
key = mx.random.key(0)
noise = mx.random.normal((2, 3), key=key)
```

Reusing the same key yields deterministic sequences.

## Randomness (MLX vs PyTorch)

- MLX uses functional, key‑based RNG similar to JAX:
  - Seed → `key = mx.random.key(seed)` → pass `key=` to ops.
  - Split with `mx.random.split(key, num)` for independent per‑sample/per‑worker streams.
- PyTorch uses an implicit global state via `torch.manual_seed` (or stateful `torch.Generator`).

Porting tips:
- Replace implicit draws with explicit keys and, when batching, split subkeys to avoid collisions.
- For vectorized code, feed a batch of subkeys into `mx.vmap(fn)` so each instance gets a unique stream.

```python
# MLX
k = mx.random.key(0)
k1, k2 = mx.random.split(k, num=2)
a = mx.random.normal(shape=(4,), key=k1)
b = mx.random.normal(shape=(4,), key=k2)

# PyTorch conceptual
# torch.manual_seed(0)
# a = torch.randn(4)
# b = torch.randn(4)
```

### Compatibility Facade (torch.Generator‑like)

If you prefer a front‑end that doesn’t expose keys at call sites, add a small wrapper that holds a seed and hands out independent draws. This mimics `torch.Generator` ergonomics while staying reproducible.

```python
import mlx.core as mx

class Generator:
    def __init__(self, seed=0):
        self.seed = int(seed); self._ctr = 0
    def _key(self):
        self._ctr += 1
        return mx.random.key(self.seed ^ (0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFF) ^ self._ctr)
    def normal(self, size, mean=0.0, std=1.0, dtype=mx.float32):
        return mx.random.normal(shape=size, loc=mean, scale=std, key=self._key()).astype(dtype)
    def uniform(self, size, low=0.0, high=1.0, dtype=mx.float32):
        return mx.random.uniform(shape=size, low=low, high=high, key=self._key()).astype(dtype)
    def randint(self, low, high, size, dtype=mx.int32):
        return mx.random.randint(low=low, high=high, shape=size, dtype=dtype, key=self._key())
    def permutation(self, n_or_x):
        return mx.random.permutation(n_or_x, key=self._key())

g = Generator(0)
x = g.normal((64, 128))
idx = g.permutation(1000)
```

Tip: You can also create a module‑level `Generator` and expose `set_seed/get_seed` for code that expects a global RNG. Prefer passing/holding a `Generator` instance explicitly in training code for clearer state.

## Optimizers

PyTorch:

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

MLX:

```python
from mlx.optimizers import AdamW
opt = AdamW(1e-3)
params = opt.update(params, grads)
```

Learning rate schedules compose directly with optimizers in MLX (see Optimizers guide).

## Common API Mappings

- Tensor to array: `torch.tensor` -> `mx.array` / `mx.asarray`
- Random: `torch.randn` -> `mx.random.normal`, `torch.rand` -> `mx.random.uniform`
- Reductions: `torch.sum` -> `mx.sum`, `torch.mean` -> `mx.mean`
- Linalg: `torch.linalg.solve` -> `mx.linalg.solve`, etc.
- NN layers: `torch.nn.Linear` -> `mlx.nn.Linear`, `Conv2d` -> `mlx.nn.Conv2d`, etc.

## Example Port: CNN Classifier

PyTorch (sketch):

```python
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 10)
)
```

MLX (sketch):

```python
net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.AvgPool2d(2), nn.AvgPool2d(2),  # use explicit pools
    nn.Flatten(), nn.Linear(64 * 8 * 8, 10)
)
```

The exact pooling strategy may differ; verify output shapes when you port.

## Takeaways

- Think functional: compute grads with transforms; pass params explicitly when helpful.
- Be explicit with shapes and devices; avoid implicit global state.
- Start with direct layer/optimizer substitutions, then refine per‑layer nuances.
