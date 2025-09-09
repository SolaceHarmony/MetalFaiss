# mlx.nn.Module vs torch.nn.Module: Practical Primer

This guide highlights how MLX modules differ from PyTorch modules and how to work productively with MLX’s functional, device‑agnostic style.

## Big Picture

- Functional style: you usually pass parameters explicitly and receive updated parameters back (via optimizers), instead of mutating a module in place.
- No `device=` keyword on ops: control placement using default device, a scoped context, or per‑op streams when supported.
- Lazy compute + unified memory: operations are recorded and executed when needed; CPU and GPU share memory on Apple Silicon.

## Parameter Handling

- MLX: parameters are discovered from arrays created inside a Module’s `__init__` and stored in a structured tree returned by `module.parameters()`.
- Torch: parameters are `nn.Parameter` attributes tracked automatically; `state_dict()` holds tensors and buffers.

Example (MLX):
```python
import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out)
        )
    def __call__(self, x):
        return self.layers(x)

net = MLP(32, 64, 10)
params = net.parameters()               # pure parameter tree
```

## Execution Model

- MLX: functional transforms (`mx.value_and_grad`, `mx.vmap`, etc.) operate on pure functions of params and inputs; optimizers return updated params.
- Torch: training typically mutates module + optimizer state in place (`loss.backward(); opt.step()`).

Training loop (MLX):
```python
from mlx.optimizers import AdamW

opt = AdamW(1e-3)

def loss_fn(p, x, y):
    logits = net.apply(p, x)     # run with explicit params
    return mx.mean((logits - y) ** 2)

for step in range(1000):
    x = mx.random.normal((64, 32))
    y = mx.random.normal((64, 10))
    loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
    params = opt.update(params, grads)
```

## State and I/O

- MLX: `module.parameters()` for params; `module.state()` for params + buffers. Save/load via `module.save_weights()`/`load_weights()` or `mx.save` for trees.
- Torch: `state_dict()` and `load_state_dict()` are the standard serialization APIs.

```python
net.save_weights("weights.npz")
net.load_weights("weights.npz")
```

## Devices and Precision

- No `device=` argument on ops:
  - Global: `mx.set_default_device(mx.cpu)` or `mx.gpu`
  - Scoped: `with mx.default_device(mx.cpu): ...`
  - Per‑op: `op(..., stream=mx.cpu)` where supported
- Float64 is CPU‑only; use CPU scopes/streams for double‑precision sections.

## Porting from PyTorch

- Keep architectures similar; change training loops to explicit param passing.
- Replace `.backward()` with `mx.value_and_grad`; replace `opt.step()` with `params = opt.update(params, grads)`.
- Remove `.to(device)` calls; instead set default device or pass `stream=mx.cpu` when needed.

## Debugging Tips

- Print shapes and keys from `params`/`grads` to catch structure mismatches early.
- Use `reduction='none'` losses to inspect per‑example values.
- If a backend errors on GPU (e.g., SVD/FFT), run that op on CPU with a stream or in a scoped block.

```python
# Inspect keys and a few shapes
print(list(params.keys()))
```

## Cheatsheet: Concept Mapping

- Torch `state_dict()` ↔ MLX `module.state()` (tree) / `module.save_weights()`
- Torch `param.to(device)` ↔ MLX default/scoped device or `stream`
- Torch `loss.backward()` ↔ MLX `mx.value_and_grad(loss_fn)`
- Torch `opt.step()` ↔ MLX `opt.update(params, grads)`

