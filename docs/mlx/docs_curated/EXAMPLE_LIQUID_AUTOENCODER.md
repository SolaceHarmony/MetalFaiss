# Ultimate MLX Patterns: Liquid Autoencoder Walkthrough (Selected Snippets)

This curated walkthrough extracts practical, runnable snippets from a comprehensive MLX example that combines compile, Module patterns, attention, quantization hooks, tree utilities, distributed setup, and Metal profiling. Use it as a pattern bank when you need to remember “how do I do X in MLX?”

Note: MLX avoids a `device=` argument on most ops. Control placement via default device, a scoped context, or per‑op `stream` (e.g., `stream=mx.cpu`). Float64 is CPU‑only.

## Compile + Metal‑Optimized Step

```python
import mlx.core as mx

@mx.compile
def compiled_ode_step(inputs, hidden, W_sensory, W_backbone, tau, dt=0.1):
    sensory = mx.sigmoid(inputs @ W_sensory) * mx.tanh(inputs @ W_backbone)
    dhdt = (-hidden + sensory) / tau
    return hidden + dt * dhdt
```

- `@mx.compile` fuses the step into a single Metal kernel when possible.
- Works with unified memory; no `.to(device)` required.

## Custom Module: Parameters as Arrays + Compiled Helper

```python
import mlx.nn as nn

class MetalOptimizedLTCCell(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.W_sensory = mx.random.normal((input_dims, hidden_dims)) * 0.1
        self.W_backbone = mx.random.normal((input_dims, hidden_dims)) * 0.1
        self.tau = mx.ones((hidden_dims,)) * 2.0
        self.ode_step = compiled_ode_step

    def __call__(self, inputs: mx.array, hidden=None):
        if hidden is None:
            hidden = mx.zeros((inputs.shape[0], self.W_sensory.shape[1]))
        h_new = self.ode_step(inputs, hidden, self.W_sensory, self.W_backbone, self.tau)
        return h_new, h_new
```

- MLX `nn.Module` discovers parameter arrays under the hood; no `nn.Parameter` wrappers.
- See Module Primer for MLX↔PyTorch differences.

## Encoder/Decoder with Attention, RoPE, and Gating

```python
class LiquidAutoencoderUltimate(nn.Module):
    def __init__(self, seq_length, input_dim=1, latent_dim=64, hidden_dim=128, num_heads=4, use_rope=True):
        super().__init__()
        self.seq_length = seq_length
        self.input_norm = nn.RMSNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = MetalOptimizedLTCCell(hidden_dim, hidden_dim)
        self.encoder_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.rope = nn.RoPE(latent_dim) if use_rope else None
        self.attention = nn.MultiHeadAttention(dims=latent_dim, num_heads=num_heads, bias=False)
        self.decoder = MetalOptimizedLTCCell(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.GLU(axis=-1)

    def encode(self, x):
        b, T, _ = x.shape
        x = self.input_norm(x)
        x = self.input_proj(x)
        h = mx.zeros((b, self.encoder.W_sensory.shape[1]))
        hs = []
        for t in range(T):
            h, _ = self.encoder(x[:, t, :], h)
            hs.append(h)
        hidden_seq = mx.stack(hs, axis=1)
        latent_seq = self.encoder_to_latent(hidden_seq)
        if self.rope:
            latent_seq = self.rope(latent_seq)
        attended = self.attention(latent_seq, latent_seq, latent_seq)
        latent = mx.mean(attended, axis=1)
        return latent, hidden_seq

    def decode(self, latent, skip_states):
        b = latent.shape[0]
        out = []
        lat = mx.repeat(latent[:, None, :], self.seq_length, axis=1)
        h = mx.zeros((b, self.decoder.W_sensory.shape[1]))
        for t in range(self.seq_length):
            h, _ = self.decoder(lat[:, t, :], h)
            if skip_states is not None and t < skip_states.shape[1]:
                combined = mx.concatenate([h, skip_states[:, t, :]], axis=-1)
                h = self.gate(combined)
            out.append(self.output_proj(h))
        return mx.stack(out, axis=1)

    def __call__(self, x):
        z, hs = self.encode(x)
        return self.decode(z, hs)
```

- Patterns: RMSNorm, RoPE, MHA, GLU gating, shape‑safe concatenation.
- Replace NumPy thinking: no `device=`, per‑op `stream` controls placement.

## Training Loop Patterns (Two Styles)

Functional style (explicit params):
```python
from mlx.optimizers import AdamW

opt = AdamW(1e-3)
params = model.parameters()

def loss_fn(p, x):
    out = model.apply(p, x)
    return mx.mean((out - x) ** 2)

loss, grads = mx.value_and_grad(loss_fn)(params, batch)
params = opt.update(params, grads)
```

Module‑aware convenience (when available in your MLX build):
```python
from mlx.optimizers import Lion, clip_grad_norm

class AdvancedTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.opt = Lion(learning_rate=lr)
        self.loss_and_grad = nn.value_and_grad(model, lambda m, x: mx.mean((m(x) - x) ** 2))

    def step(self, batch):
        loss, grads = self.loss_and_grad(self.model, batch, batch)
        grads, _ = clip_grad_norm(grads, max_norm=1.0)  # returns (grads, norm)
        self.opt.update(self.model, grads)
        return loss
```

- Remember: `clip_grad_norm` returns `(clipped_grads, total_norm)`; unpack before updating.

## Quantization Hook (Concept)

```python
def quantize_model(self, bits=4, group_size=64):
    # nn.quantize(self, bits=bits, group_size=group_size)
    # Provide weights in expected format before calling; omitted here.
    pass
```

- Use quantization to reduce memory for inference; measure accuracy impact.

## Tree Utilities for Checkpointing

```python
from mlx.utils import tree_flatten

flat = tree_flatten(model.parameters())
# Convert to key->array dict and save
save_dict = {".".join(map(str, k)) if isinstance(k, (list, tuple)) else str(k): v for k, v in flat}
mx.save_safetensors("model.safetensors", save_dict)
```

- Rebuild nested trees from flat keys when loading; update via `model.update(nested_params)`.

## Distributed (Optional)

```python
try:
    import mlx.core.distributed as dist
    if dist.is_available():
        dist.init()
        print("Distributed rank", dist.get_rank(), "/", dist.get_world_size())
except Exception:
    pass
```

- Initialize if available; design your script to still run single‑process.

## Metal Profiling (Optional)

```python
if mx.metal.is_available():
    try:
        mx.metal.start_capture("profile.gputrace")
        _ = model(x)
        mx.eval(_)
        mx.metal.stop_capture()
    except RuntimeError:
        pass
```

- Use capture around a tiny, representative region to keep traces small.

## Device & Precision Reminders

- No `device=` keyword: control via `mx.set_default_device`, `with mx.default_device(...)`, or per‑op `stream`.
- Float64 is CPU‑only; scope double‑precision computations to CPU.

For deeper differences in Module behavior vs PyTorch, see: `MODULE_PRIMER.md`.

