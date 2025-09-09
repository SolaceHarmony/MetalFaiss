# Neural Networks with MLX (`mlx.nn`)

MLX provides a lightweight, composable neural network library with a `Module` base class, common layers, activations, and utilities.

## Core Ideas

- `nn.Module`: define parameters and computation by composing modules.
- Functional style: `module.apply(params, x)` lets you run with an explicit parameter set.
- Training loop: compute loss, get gradients with `mx.value_and_grad`, update with an optimizer.

## Defining Modules

```python
import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_out)
        )
    def __call__(self, x):
        return self.layers(x)

model = MLP(32, 64, 10)
params = model.parameters()
```

Useful APIs:

- `module.parameters()`: returns parameter tree
- `module.state()`: returns full state (params + buffers)
- `module.apply(params, *args)`: call with explicit params
- `module.train()` / `module.eval()`: set training mode

## Common Layers

- Linear / Conv1d/2d/3d / ConvTranspose1d/2d/3d
- Pooling: AvgPool1d/2d/3d, MaxPool1d/2d/3d
- Norms: BatchNorm, LayerNorm, GroupNorm, RMSNorm, InstanceNorm
- Attention: MultiHeadAttention, RoPE, ALiBi
- Positional encodings: SinusoidalPositionalEncoding
- Activations: ReLU, LeakyReLU, GELU (and variants), ELU, SiLU, Softplus, Softshrink, LogSigmoid, Tanh, etc.

Example:

```python
net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 10)
)
```

## Losses

Available in `nn.losses`: cross entropy, MSE, L1, KL‑div, Huber, margin ranking, triplet loss, NLL, binary cross‑entropy, etc.

```python
from mlx.nn import losses

def loss_fn(params, x, y):
    logits = net.apply(params, x)
    return losses.cross_entropy(logits, y)
```

## Training Loop

```python
import mlx.core as mx
from mlx.optimizers import Adam

params = net.parameters()
opt = Adam(1e-3)

for step in range(1000):
    x = mx.random.normal((64, 3, 32, 32))
    y = mx.random.randint(0, 10, (64,))
    l, grads = mx.value_and_grad(loss_fn)(params, x, y)
    params = opt.update(params, grads)
    if step % 100 == 0:
        print(step, l.item())
```

## Quantization

Selected layers have quantized variants (e.g., `QuantizedLinear`, `QuantizedEmbedding`). Use them to reduce memory/computation; evaluate accuracy impact.

## Tips

- Call `module.train()` for training and `module.eval()` for evaluation to toggle normalization/dropout behavior.
- Use `Module.save_weights()` and `Module.load_weights()` for lightweight checkpoints, or global save/load for full state.
- Consider `nn.Sequential` for simple feed‑forward stacks; write custom `__call__` for complex architectures.

## Module vs PyTorch (Quick)

- No `device=` on ops; control placement via default device/`stream`.
- Parameters are returned as a tree (`module.parameters()`); optimizers return updated trees instead of mutating in place.
- Replace `.backward()` with `mx.value_and_grad`, and `opt.step()` with `params = opt.update(params, grads)`.

See: `../docs_curated/MODULE_PRIMER.md` for a deeper comparison and examples.
