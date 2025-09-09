Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (fft.md):
- API index for FFT variants; lots of Sphinx wrappers.
- Users need quick guidance on axes, real vs complex, and shape expectations.
-->

## Curated Notes

- Real transforms (`rfft*`) produce complex outputs with Hermitian symmetry; match `irfft*` shapes accordingly (`n` controls length).
- Multi‑axis transforms: specify `axes=(...)` explicitly to avoid surprises.
- Use `.real`/`.imag` to extract components; keep dtypes consistent when post‑processing.

Backend note: Some builds support FFT on CPU only. If your GPU backend errors, run FFTs on the CPU using the `stream` parameter (or a CPU device scope) and keep the rest of the pipeline on GPU. MLX ops generally do not take a `device=` parameter; use default device/streams instead.

### CPU Stream Example

```python
import mlx.core as mx

# Unified memory: accessible by both CPU and GPU
a = mx.random.uniform(shape=(16,))

# Perform FFT and iFFT explicitly on CPU
fft_cpu = mx.fft.fft(a, stream=mx.cpu)
ifft_cpu = mx.fft.ifft(fft_cpu, stream=mx.cpu)
```

### Global Default Device Example

```python
import mlx.core as mx

# Route subsequent ops to CPU globally
mx.set_default_device(mx.cpu)

a = mx.random.uniform(shape=(16,))
fft_result = mx.fft.fft(a)   # runs on CPU due to default device

# Reset later if desired
# mx.set_default_device(mx.gpu)
```

## Potential Issues

- Mixed‑device streams: If you use both CPU and GPU streams, MLX tracks dependencies. When a GPU op consumes results from a CPU op (or vice‑versa), MLX ensures the producer completes first. You generally don’t need to add manual barriers beyond `mx.synchronize()` at host boundaries.
- Performance implications: GPUs typically accelerate large FFTs substantially. For small inputs, CPU can be competitive and unified memory helps avoid transfer overhead. Benchmark both paths at your problem sizes.
- Numerical stability: Certain FFT‑based pipelines may exhibit numerical instability in specific configurations. If you see unexpected results, try alternate precisions (e.g., float32), different batching/tiling, or switch device backends to isolate the effect while MLX improvements land.

### Examples

```python
import mlx.core as mx

# 1D FFT and inverse
sig = mx.random.normal((1024,))
spec = mx.fft.fft(sig)
recon = mx.fft.ifft(spec).real

# Real-input transforms
r = mx.random.normal((256,))
R = mx.fft.rfft(r)
r_back = mx.fft.irfft(R, n=r.shape[0])

# 2D FFT over specified axes
img = mx.random.normal((64, 64))
F = mx.fft.fft2(img)
img_back = mx.fft.ifft2(F).real

# N-D with custom axes
vol = mx.random.normal((8, 16, 32))
G = mx.fft.fftn(vol, axes=(1, 2))
```


<div id="main-content" class="bd-main" role="main">

<div class="sbt-scroll-pixel-helper">

</div>

<div class="bd-content">

<div class="bd-article-container">

<div class="bd-header-article d-print-none">

<div class="header-article-items header-article__inner">

<div class="header-article-items__start">

<div class="header-article-item">

<span class="fa-solid fa-bars"></span>

</div>

</div>

<div class="header-article-items__end">

<div class="header-article-item">

<div class="article-header-buttons">

<a href="https://github.com/ml-explore/mlx"
class="btn btn-sm btn-source-repository-button"
data-bs-placement="bottom" data-bs-toggle="tooltip" target="_blank"
title="Source repository"><span class="btn__icon-container"> <em></em>
</span></a>

<div class="dropdown dropdown-download-buttons">

- <a
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/fft.rst"
  class="btn btn-sm btn-download-source-button dropdown-item"
  data-bs-placement="left" data-bs-toggle="tooltip" target="_blank"
  title="Download source file"><span class="btn__icon-container">
  <em></em> </span> <span class="btn__text-container">.rst</span></a>
- <span class="btn__icon-container"> </span>
  <span class="btn__text-container">.pdf</span>

</div>

<span class="btn__icon-container"> </span>

</div>

</div>

</div>

</div>

</div>

<div id="jb-print-docs-body" class="onlyprint">

# FFT

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="fft" class="section">

<span id="id1"></span>

# FFT<a href="https://ml-explore.github.io/mlx/build/html/#fft"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft.html#mlx.core.fft.fft"
class="reference internal" title="mlx.core.fft.fft"><span
class="pre"><code class="sourceCode python">fft</code></span></a>(a\[, n, axis, stream\]) | One dimensional discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifft.html#mlx.core.fft.ifft"
class="reference internal" title="mlx.core.fft.ifft"><span
class="pre"><code class="sourceCode python">ifft</code></span></a>(a\[, n, axis, stream\]) | One dimensional inverse discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft2.html#mlx.core.fft.fft2"
class="reference internal" title="mlx.core.fft.fft2"><span
class="pre"><code class="sourceCode python">fft2</code></span></a>(a\[, s, axes, stream\]) | Two dimensional discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifft2.html#mlx.core.fft.ifft2"
class="reference internal" title="mlx.core.fft.ifft2"><span
class="pre"><code class="sourceCode python">ifft2</code></span></a>(a\[, s, axes, stream\]) | Two dimensional inverse discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftn.html#mlx.core.fft.fftn"
class="reference internal" title="mlx.core.fft.fftn"><span
class="pre"><code class="sourceCode python">fftn</code></span></a>(a\[, s, axes, stream\]) | n-dimensional discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifftn.html#mlx.core.fft.ifftn"
class="reference internal" title="mlx.core.fft.ifftn"><span
class="pre"><code class="sourceCode python">ifftn</code></span></a>(a\[, s, axes, stream\]) | n-dimensional inverse discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft.html#mlx.core.fft.rfft"
class="reference internal" title="mlx.core.fft.rfft"><span
class="pre"><code class="sourceCode python">rfft</code></span></a>(a\[, n, axis, stream\]) | One dimensional discrete Fourier Transform on a real input. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfft.html#mlx.core.fft.irfft"
class="reference internal" title="mlx.core.fft.irfft"><span
class="pre"><code class="sourceCode python">irfft</code></span></a>(a\[, n, axis, stream\]) | The inverse of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft.html#mlx.core.fft.rfft"
class="reference internal" title="mlx.core.fft.rfft"><span
class="pre"><code class="sourceCode python">rfft()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft2.html#mlx.core.fft.rfft2"
class="reference internal" title="mlx.core.fft.rfft2"><span
class="pre"><code class="sourceCode python">rfft2</code></span></a>(a\[, s, axes, stream\]) | Two dimensional real discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfft2.html#mlx.core.fft.irfft2"
class="reference internal" title="mlx.core.fft.irfft2"><span
class="pre"><code class="sourceCode python">irfft2</code></span></a>(a\[, s, axes, stream\]) | The inverse of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft2.html#mlx.core.fft.rfft2"
class="reference internal" title="mlx.core.fft.rfft2"><span
class="pre"><code class="sourceCode python">rfft2()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfftn.html#mlx.core.fft.rfftn"
class="reference internal" title="mlx.core.fft.rfftn"><span
class="pre"><code class="sourceCode python">rfftn</code></span></a>(a\[, s, axes, stream\]) | n-dimensional real discrete Fourier Transform. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfftn.html#mlx.core.fft.irfftn"
class="reference internal" title="mlx.core.fft.irfftn"><span
class="pre"><code class="sourceCode python">irfftn</code></span></a>(a\[, s, axes, stream\]) | The inverse of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfftn.html#mlx.core.fft.rfftn"
class="reference internal" title="mlx.core.fft.rfftn"><span
class="pre"><code class="sourceCode python">rfftn()</code></span></a>. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.fast.metal_kernel

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.fft.fft

</div>

</div>

</div>

</div>

<div class="bd-footer-content__inner container">

<div class="footer-item">

By MLX Contributors

</div>

<div class="footer-item">

© Copyright 2023, MLX Contributors.  

</div>

<div class="footer-item">

</div>

<div class="footer-item">

</div>

</div>

</div>
