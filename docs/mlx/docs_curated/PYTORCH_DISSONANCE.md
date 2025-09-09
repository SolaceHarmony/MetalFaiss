# PyTorch Muscle Memory vs MLX: Dissonance Checklist

A quick-reference for patterns that differ between PyTorch and MLX. If your intuition says “like PyTorch,” verify here first.

## Autograd and Updates

- No `.backward()` side effects; use `mx.value_and_grad(fn)` to get gradients.
- No `.step()`/`.zero_grad()`; optimizers return new params: `params = opt.update(params, grads)`.
- Gradients are plain trees matching `params` shape; keep structures identical.

## In‑Place vs Functional

- Avoid in‑place mutations; MLX APIs are functional and favor returning new arrays.
- Use `mx.where` and pure ops; don’t rely on `x.add_()`‑style in‑place updates.

## Devices

- MLX chooses Metal GPU by default on Apple Silicon. Scope with `with mx.default_device(...)` for CPU/GPU.
- No ubiquitous `.to(device)`; placement is controlled via default device and execution context.

## Execution Model

- PyTorch is eager by default; MLX is lazy. Most ops build a graph and run at eval points.
- Force compute with `mx.eval(x, ...)` or `.item()` on scalars. Profile around evals, not line‑by‑line.

## Performance Across Apple Silicon Generations

- Expect variance by chip and workload:
  - M1/M2: MLX can outperform PyTorch MPS on many linalg/elementwise tasks.
  - M3: PyTorch MPS often does very well on conv‑heavy models (e.g., ResNet training); MLX may still lead on some kernels.
- Benchmark your exact model/training loop; prefer compiled, pure MLX graphs to maximize fusion.

## Model Conversion and Ecosystem

- PyTorch benefits from a large ecosystem; models from Hugging Face are usually PyTorch‑ready.
- With MLX you often convert weights first:
  - Load PyTorch weights → export (safetensors/npz) → construct MLX module → load weights.
  - See Save/Load guide and MLX examples for scripts/utilities.

## Debugging Lazy Evaluation

- Errors may be raised at eval time, not at the line constructing the graph.
- Tactics:
  - Insert `mx.eval(...)` at boundaries to pinpoint failing segments.
  - Print shapes and a few sample values (via `.item()`/small slices) around suspected edges.
  - Keep side‑effects out of loss/grad code; stick to pure MLX ops.

## Optimizer State and Updates

- PyTorch optimizers are stateful and in‑place; MLX optimizers return new params (and manage their internal state).
- Pattern:
  ```python
  value, grads = mx.value_and_grad(loss_fn)(params, batch)
  params = opt.update(params, grads)
  mx.eval(params)  # force if you need updated arrays now
  ```

## Portability and Deployment

- PyTorch spans CUDA, CPU, MPS and many deployment targets.
- MLX is Apple‑only; code won’t run on non‑Apple GPUs without porting.

## API Differences (Utility Functions)

- Parameter names may differ: MLX often uses `axis` where PyTorch uses `dim`.
- `keepdims/keepdim`: supported in many reductions but not universal; check signatures.
- Favor explicit `axis=` and review docstrings when porting.

See also:
- Porting from PyTorch to MLX
- Backend Support Notes (CPU vs GPU)
- Saving and Loading (conversion checklist)
- Random (MLX vs PyTorch RNG)

## Cheat Sheet (PyTorch → MLX)

- `loss.backward()` → `value, grads = mx.value_and_grad(loss_fn)(params, batch)`
- `optimizer.step()` → `params = opt.update(params, grads)`
- `optimizer.zero_grad()` → not needed (no tensor‑attached grads)
- `tensor.to(device)` / `.cuda()` / `.cpu()` → `mx.set_default_device(...)` or `with mx.default_device(...)`
- `torch.manual_seed(n)` → `key = mx.random.key(n)`; split: `k1, k2 = mx.random.split(key, num=2)`
- `torch.Generator` → small wrapper that hands out keys internally (see Random doc)
- `x[mask] = v` → `mx.where(mask, v, x)` (returns new array)
- `x[idx] = v` (slice) → `x.at[idx].set(v)` or `mx.slice_update(...)`
- `dim=` / `keepdim=` → `axis=` / `keepdims=` (when supported)
- `state_dict()` / `load_state_dict()` → `save_weights()/load_weights()` or `mx.save/mx.load`
- `.item()` logging → same; also triggers evaluation

## Indexing and Updates

- No boolean‑mask assignment like `x[mask] = v`; use `mx.where(mask, v, x)`.
- Use `array.at[...]` or `mx.slice_update` for slice/index replacement; arrays are immutable.

## Views

- Slicing returns strided views, but no in‑place ops exist to mutate backing storage. This avoids overlapping‑view hazards common in PyTorch.

## RNG

- Prefer explicit keys: `key = mx.random.key(seed)`; pass `key=` into random ops for reproducibility.
- Reuse keys for deterministic sequences; don’t expect a single global seed to govern all ops.

## Shapes and Layout

- Convolutions assume `(N, C, H, W)`; validate output math explicitly.
- Broadcasting is NumPy‑like; be explicit with `keepdims` and reshapes to avoid silent mismatches.

## Losses

- Losses live under `mlx.nn.losses`; signatures mirror NumPy/JAX‑style. Ensure your loss reduces to a scalar.
- Classification: pass integer class indices where applicable (not one‑hot) unless the API specifies otherwise.

## Optimizers

- Pass a scalar `lr` or a schedule function. No optimizer param groups by default; use schedules or multiple optimizers for different subtrees.
- `clip_grad_norm` returns `(clipped, total_norm)`; unpack before `opt.update`.

## State and Checkpointing

- `module.parameters()` returns a pure param tree; `module.state()` adds buffers.
- Use `module.save_weights()/load_weights()` for module weights, or `mx.save/mx.load` for arbitrary trees.

## Debugging Patterns

- Print tree keys and a few shapes when errors arise.
- Convert scalars via `.item()` for logging; beware of non‑contiguous views in heavy kernels (use `mx.contiguous`).
