# Optimizers (`mlx.optimizers`)

MLX includes practical optimizers and schedules to train models with minimal boilerplate.

## Basic Usage

```python
from mlx.optimizers import Adam, SGD, RMSprop, AdamW

opt = Adam(1e-3)
params = model.parameters()

for step in range(1000):
    loss, grads = mx.value_and_grad(loss_fn)(params, batch_x, batch_y)
    params = opt.update(params, grads)
```

## Available Optimizers

- SGD (+ momentum via kwargs)
- Adam / AdamW / Adamax / RMSprop / Adagrad / AdaDelta / Adafactor / Lion

Each optimizer exposes `init`, `state`, and `update` under the hood; the standard `.update(params, grads)` is the ergonomic entry.

## Schedules

Learning rate schedules compose with optimizers:

```python
from mlx.optimizers import cosine_decay, exponential_decay, step_decay, linear_schedule, join_schedules

schedule = join_schedules([
    linear_schedule(0.0, 1e-3, warmup_steps=100),
    cosine_decay(1e-3, total_steps=10000),
])

opt = Adam(schedule)
```

## Gradient Utilities

- `clip_grad_norm(params_or_grads, max_norm) -> (clipped, total_norm)`: prevent exploding gradients.

Example:

```python
from mlx.optimizers import clip_grad_norm

loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
grads, total_norm = clip_grad_norm(grads, 1.0)
params = opt.update(params, grads)
```

## Tips

- Prefer AdamW for transformer‑like models; SGD/momentum for CNNs can be strong baselines.
- Use schedules for long runs; warmup avoids early instability, cosine decay provides smooth landing.
- Keep an eye on weight decay settings; for AdamW it’s decoupled (preferred for many architectures).
