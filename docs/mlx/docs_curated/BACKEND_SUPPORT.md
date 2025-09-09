# Backend Support Notes (CPU vs GPU)

This page summarizes MLX operations that commonly run CPU‑only in current releases/builds and provides guidance for handling them. Exact coverage evolves—always prefer empirical checks in your environment.

## Quick Guidance

- Prefer writing code that is device‑agnostic; add small, local CPU scopes around ops that fail on your GPU backend.
- Patterns:
  ```python
  import mlx.core as mx
  with mx.default_device(mx.cpu):
      out = some_op(x)
  ```
- When the op supports it, pass a CPU stream/device directly:
  ```python
  out = some_op(x, stream=mx.cpu)
  ```
- To route all subsequent ops to CPU, change the default device:
  ```python
  mx.set_default_device(mx.cpu)
  # ... ops here run on CPU by default
  # mx.set_default_device(mx.gpu)  # switch back when needed
  ```
- Keep tensors on the default device elsewhere; only scope to CPU for the minimal region.

## Apple Silicon Performance Notes (M1/M2/M3)

- Performance varies by chip and workload:
  - M1/M2: MLX often excels at linear algebra and data‑parallel elementwise ops; many users report strong throughput.
  - M3: PyTorch MPS may outperform MLX on conv‑heavy training (e.g., ResNet) in some builds; MLX may still shine on certain linalg or sorting tasks.
- GPU clocks and driver behavior can differ by stack; always profile your exact workload on your machine.
- Practical advice:
  - Prefer MLX when you need functional transforms, custom Metal kernels, or you’re linalg‑bound.
  - For conv‑heavy baselines where MPS is known to be strong, benchmark PyTorch MPS vs MLX before committing.
  - Route float64 to CPU; consider mixed precision for throughput.
  - Use `mx.compile` (if available) and keep graphs pure to maximize fusion.

See also:
- PyTorch Muscle Memory vs MLX (Dissonance)
- Porting from PyTorch to MLX
- Random (RNG keys vs global state)

## Likely CPU‑Only (by category)

- Matrix/Linear Algebra:
  - QR decomposition: `mx.linalg.qr`
  - SVD: `mx.linalg.svd`
  - Eigendecomposition: `mx.linalg.eig` / `mx.linalg.eigh` / `mx.linalg.eigvalsh`
  - LU: `mx.linalg.lu` / `mx.linalg.lu_factor`
  - Cholesky: `mx.linalg.cholesky` (+ `cholesky_inv`)
  - Inverse: `mx.linalg.inv` / `mx.linalg.tri_inv`
- Array/Matrix Manipulation:
  - Quantized matmul: `mx.quantized_matmul`, `mx.quantize` (and related `fast` APIs)
  - Block/Segment ops: `mx.block_masked_mm`, `mx.segmented_mm`
  - Gather variants: `mx.gather_mm`, `mx.gather_qmm`
- FFT/Transforms:
  - FFT family: `mx.fft.fft`, `ifft`, `fftn`, `ifftn`, `fft2`, `ifft2`, `rfft*`, `irfft*`
- Elementwise/Reductions (selected):
  - Scans/reduces: `mx.scan`, `mx.reduce`
  - Sorting/arg ops: `mx.sort`, `mx.argsort`, `mx.argpartition`, `mx.argmax/argmin` (backends vary)
  - Softmax: `mx.softmax` (use CPU scope if your backend errs)
- Slicing and Indexing:
  - `mx.gather`, `mx.gather_axis`, `mx.slice`, `mx.slice_update`, `mx.put_along_axis`
- Utilities:
  - `mx.partition`, `mx.pad`, `mx.view`, `mx.unflatten`, `mx.flatten`, `mx.squeeze`, `mx.expand_dims`, `mx.reshape`, `mx.contiguous`

Notes:
- The above is intentionally conservative. Some builds may support subsets on GPU; test locally.
- When an op lacks GPU support, MLX may raise a backend error—scope that op to CPU and proceed.

## Testing Support Quickly

```python
import mlx.core as mx

def supported_on(device, fn, *args, **kwargs):
    try:
        with mx.default_device(device):
            _ = fn(*args, **kwargs)
        return True
    except Exception:
        return False

A = mx.random.normal((8, 4))
print("SVD on GPU:", supported_on(mx.gpu, mx.linalg.svd, A))
print("SVD on CPU:", supported_on(mx.cpu, mx.linalg.svd, A))
```

Keep this probe local and fast; cache results for your run rather than probing repeatedly.
