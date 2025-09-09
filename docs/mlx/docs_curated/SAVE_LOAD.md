# Saving and Loading

MLX provides simple file I/O for arrays and model weights.

## Arrays and Trees

```python
import mlx.core as mx

data = {
  "x": mx.arange(10),
  "batch": [mx.ones((2, 3)), mx.zeros((2, 3))],
}

mx.save("data.npz", data)
restored = mx.load("data.npz")
```

Compressed variants are available via `savez`/`savez_compressed`.

## Safetensors and GGUF

For tensor‑only formats:

```python
mx.save_safetensors("weights.safetensors", model.parameters())
```

For GGUF (useful for LLM tooling):

```python
mx.save_gguf("model.gguf", model.state())
```

## Module Weights API

```python
import mlx.nn as nn

model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 10))

model.save_weights("weights.npz")
model.load_weights("weights.npz")
```

Prefer `save_weights`/`load_weights` for pure module parameters, and `mx.save`/`mx.load` for arbitrary trees.

## Converting PyTorch Weights → MLX (Checklist)

- Export from PyTorch:
  - Load the PyTorch model, call `state_dict()`.
  - Convert tensors to NumPy (or save `safetensors`).
  - Optional: write a small mapping if parameter names differ.
- Construct the MLX module with matching shapes/layouts.
- Load weights:
  - If using `npz`/dict: convert arrays to MLX via `mx.array` and assign into the parameter tree, or write an MLX tree and `mx.save`/`mx.load`.
  - If using `safetensors`: `mx.load_safetensors` (if available) or load arrays and build the tree.

Minimal example (npz route):
```python
# PyTorch side (export)
import torch, numpy as np
sd = torch.load("pytorch_model.pth", map_location="cpu")
np_sd = {k: v.detach().cpu().numpy() for k, v in sd.items()}
np.savez("weights_pt.npz", **np_sd)

# MLX side (import)
import mlx.core as mx
import mlx.nn as nn
from collections import OrderedDict

model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 10))
pt = dict(np.load("weights_pt.npz"))

# Map names if needed (example only)
name_map = {
  "layers.0.weight": "layers.0.weight",
  "layers.0.bias": "layers.0.bias",
  "layers.2.weight": "layers.2.weight",
  "layers.2.bias": "layers.2.bias",
}

params = model.parameters()
for k_pt, k_mx in name_map.items():
    arr = mx.array(pt[k_pt])
    # assign into the params tree; use your tree utilities to match paths
    # e.g., params = set_in_tree(params, k_mx.split('.'), arr)

# Or rebuild a mirrored tree then `model.load_weights` from a saved file
```

Tips:
- Pay attention to layout differences (e.g., PyTorch Linear is out×in; ensure MLX modules match expected shape).
- Validate with a forward pass on the same input and compare outputs.
- See also: Porting from PyTorch (mental model), Backend Support (CPU/GPU coverage), Random (RNG differences).

## For NumPy Users

- `numpy.savez`/`savez_compressed` -> `mx.save`/`mx.savez`/`mx.savez_compressed`
- `numpy.load` -> `mx.load` (returns a Python tree of MLX arrays)
- MLX extras: `mx.save_safetensors` for tensor‑only weights; `mx.save_gguf` for GGUF model export.

See also:
- Porting from PyTorch to MLX
- PyTorch Muscle Memory vs MLX (Dissonance)
- Backend Support Notes (CPU vs GPU)
