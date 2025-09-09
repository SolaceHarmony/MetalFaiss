# FFT

Fast Fourier Transforms for 1D/2D/N‑D signals.

Key points
- Dtype/device: complex dtypes flow through; float64 typically runs on CPU. Route via `with mx.default_device(mx.cpu)` or `stream=mx.cpu`.
- Axes and length: pass `axis=` to choose dimension; pass `n=` (or `s=` for N‑D) to pad/truncate.
- Real transforms: use `rfft`/`irfft` pairs for real inputs to save work and memory.
- Batching: FFTs apply independently along batch dimensions; shapes broadcast as in NumPy.

## 1D

```python
import mlx.core as mx

signal = mx.random.normal((1024,))
spec = mx.fft.fft(signal)
recon = mx.fft.ifft(spec)

# Choose axis and pad/truncate to length n
sig2 = mx.random.normal((8, 64))
F = mx.fft.fft(sig2, n=128, axis=1)
```

## 2D (Images)

```python
img = mx.random.normal((256, 256))
freq = mx.fft.fft2(img)
img_back = mx.fft.ifft2(freq).real

# Real transforms (use rfft/irfft)
r = mx.random.normal((256, 256))
R = mx.fft.rfft2(r)
r_back = mx.fft.irfft2(R, s=r.shape)
```

## N‑D

```python
vol = mx.random.normal((64, 64, 64))
freq = mx.fft.fftn(vol)
vol_back = mx.fft.ifftn(freq).real
```

Tips:

- Use power‑of‑two sizes for best performance.
- Centering and windowing are application‑dependent; MLX exposes raw FFTs.
- Frequency indexing follows NumPy conventions; use fftshift/ifftshift if available or implement via indexing.
- Float64 route to CPU; keep the rest of the pipeline on GPU as needed.
- In pipelines, consider windowing to reduce spectral leakage; multiply by a window before FFT.

CPU routing example (double precision):
```python
with mx.default_device(mx.cpu):
    x64 = mx.linspace(0.0, 1.0, 4096).astype(mx.float64)
    X64 = mx.fft.fft(x64)
```

## For NumPy Users

- `numpy.fft.fft/ifft` -> `mx.fft.fft/ifft`
- `numpy.fft.fft2/ifft2` -> `mx.fft.fft2/ifft2`
- `numpy.fft.fftn/ifftn` -> `mx.fft.fftn/ifftn`
- Complex dtype handling mirrors NumPy; use `.real`/`.imag` as needed.
 - Real FFTs: `numpy.fft.rfft/irfft` -> `mx.fft.rfft/irfft`; ensure `s=` matches the original real shape on inverse.
