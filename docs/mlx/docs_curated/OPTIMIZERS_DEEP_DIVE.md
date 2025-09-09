# Optimizers Deep Dive (MLX)

This guide explains optimizer state, parameter tree alignment, gradient preprocessing, and schedules in MLX, with troubleshooting notes for common runtime errors.

## Parameter Trees and Optimizer State

- Parameters are nested dict/tuple structures produced by `module.parameters()`.
- Optimizers maintain internal state aligned to the parameter tree (e.g., moments for AdamW).
- Updates are pure: `new_params = opt.update(params, grads)`; the optimizer returns updated parameters.

Invariant: shapes and tree structure of `grads` must match `params` exactly.

```python
loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
assert grads.keys() == params.keys()         # top‑level check
```

If you manipulate grads, preserve structure and dtypes.

## Clip Grad Norm: Return Type

`clip_grad_norm` returns a pair `(clipped_grads, total_norm)`.

```python
from mlx.optimizers import AdamW, clip_grad_norm

opt = AdamW(1e-3)
loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
# IMPORTANT: unpack the tuple
grads, total_norm = clip_grad_norm(grads, 1.0)
params = opt.update(params, grads)
```

Passing the raw tuple to `update` will break tree walking and can yield errors like "str does not support item assignment".

## Empty Subtrees (e.g., RoPE, Gates)

Empty dicts are allowed. Optimizers skip them gracefully as long as the grad tree also has empty dicts at the same paths.

```python
params = {"linear": {...}, "rope": {}}
loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
# grads["rope"] should also be {}
params = opt.update(params, grads)
```

If a subtree has parameters but grads are missing (or vice versa), the update will fail. Ensure your loss flows through all parameterized paths you expect.

## State Initialization and Shape Mismatches

Optimizers lazily initialize state with the first update. If a layer’s parameter shape changes between steps (e.g., a dynamic module reconfiguration), state reuse will break. Keep shapes stable across steps or recreate the optimizer when you change model structure.

Checklist:
- Assert `tree_map(type, params) == tree_map(type, grads)` shapes and dtypes.
- Ensure the loss is a scalar (reduces over batch/time) so gradient shapes are well‑formed.
- Do not mix Python lists and tuples in trees across steps; keep structures consistent.

## Schedules and Warmup

All schedules produce a callable learning rate given the step. Compose with optimizers directly:

```python
from mlx.optimizers import AdamW, join_schedules, linear_schedule, cosine_decay

schedule = join_schedules([
    linear_schedule(0.0, 3e-4, warmup_steps=500),
    cosine_decay(3e-4, total_steps=100_000),
])
opt = AdamW(schedule)
```

You may also implement custom schedules by passing a function `step -> lr`.

## Gradient Transforms and Preprocessing

Common patterns before update:

```python
# 1) Global norm clipping
grads, total_norm = clip_grad_norm(grads, max_norm=1.0)

# 2) Masking (example: freeze submodule)
from mlx.utils import tree_map_with_path
mask = {"encoder": False, "head": True}

def maybe_zero(path, g):
    # path is a tuple of keys; pick a convention for your project
    if path and path[0] in mask and not mask[path[0]]:
        return mx.zeros_like(g)
    return g

grads = tree_map_with_path(maybe_zero, grads)

# 3) Mixed precision: maintain grads in param dtype to avoid casts
from mlx.utils import tree_map
grads = tree_map(lambda p, g: g.astype(p.dtype), params, grads)
```

## Troubleshooting Examples

- Error: "str does not support item assignment"
  - Cause: passing `(grads, total_norm)` tuple into `update`; unpack the tuple.

- Error: "missing key ... in optimizer state" or shape mismatch
  - Cause: param/grads trees differ or shapes changed across steps. Fix structure, or re‑create optimizer after model structure changes.

- Silent wrong updates
  - Cause: broadcasting/shape mistakes before loss reduction. Verify loss is scalar and check each axis explicitly.

## Minimal Repro Template

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW, clip_grad_norm

net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 10))
params = net.parameters()
opt = AdamW(1e-3)

def loss_fn(p, x, y):
    logits = net.apply(p, x)
    return mx.mean((logits - y) ** 2)

x = mx.random.normal((128, 32))
y = mx.random.normal((128, 10))

loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
grads, _ = clip_grad_norm(grads, 1.0)
params = opt.update(params, grads)
```

When debugging, print tree keys and a few shapes:

```python
print("Params:", list(params.keys()))
print("Grads:", list(grads.keys())))
```

This deep dive complements the Optimizers overview; use it when you need precise control over update mechanics and state.

