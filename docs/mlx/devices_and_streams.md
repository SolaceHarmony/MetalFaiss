Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (devices_and_streams.md):
- API table for devices/streams; heavy Sphinx HTML.
- Users often misinterpret default device/stream scoping and forget to synchronize when reading on host.
-->

## Curated Notes

- Inspect placement: `print(mx.default_device())`; scope with `with mx.default_device(mx.cpu): ...`.
- Stream contexts: `with mx.new_stream(): ...` queue work; call `mx.synchronize()` before consuming results on host.
- Prefer simple single‑stream usage first; add streams after profiling shows contention.
- Note: MLX ops generally do not accept a `device=` keyword. Use the default device, a scoped `with mx.default_device(...)`, a per‑op `stream=...`, or `mx.set_default_device(...)` to control placement.

### Examples

### Examples

```python
import mlx.core as mx

print(mx.default_device())

# Run a block on CPU explicitly
with mx.default_device(mx.cpu):
    a = mx.ones((1024, 1024))

# Queue work on a new stream and sync before host use
s = mx.new_stream()
with s:
    b = mx.random.normal((2048, 2048)) @ mx.random.normal((2048, 2048))
# Change default device globally (all subsequent ops)
mx.set_default_device(mx.cpu)

# Ensure work completed before converting to Python types / printing
mx.synchronize()
print(b.shape)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/devices_and_streams.rst"
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

# Devices and Streams

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="devices-and-streams" class="section">

<span id="id1"></span>

# Devices and Streams<a
href="https://ml-explore.github.io/mlx/build/html/#devices-and-streams"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html#mlx.core.Device"
class="reference internal" title="mlx.core.Device"><span
class="pre"><code class="sourceCode python">Device</code></span></a> | A device to run operations on. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre"><code class="sourceCode python">Stream</code></span></a> | A stream for running operations on a given device. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.default_device.html#mlx.core.default_device"
class="reference internal" title="mlx.core.default_device"><span
class="pre"><code
class="sourceCode python">default_device</code></span></a>() | Get the default device. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_default_device.html#mlx.core.set_default_device"
class="reference internal" title="mlx.core.set_default_device"><span
class="pre"><code
class="sourceCode python">set_default_device</code></span></a>(device) | Set the default device. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.default_stream.html#mlx.core.default_stream"
class="reference internal" title="mlx.core.default_stream"><span
class="pre"><code
class="sourceCode python">default_stream</code></span></a>(device) | Get the device's default stream. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.new_stream.html#mlx.core.new_stream"
class="reference internal" title="mlx.core.new_stream"><span
class="pre"><code class="sourceCode python">new_stream</code></span></a>(device) | Make a new stream on the given device. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_default_stream.html#mlx.core.set_default_stream"
class="reference internal" title="mlx.core.set_default_stream"><span
class="pre"><code
class="sourceCode python">set_default_stream</code></span></a>(stream) | Set the default stream. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stream.html#mlx.core.stream"
class="reference internal" title="mlx.core.stream"><span
class="pre"><code class="sourceCode python">stream</code></span></a>(s) | Create a context manager to set the default device and stream. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html#mlx.core.synchronize"
class="reference internal" title="mlx.core.synchronize"><span
class="pre"><code
class="sourceCode python">synchronize</code></span></a>(\[stream\]) | Synchronize with the given stream. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.finfo.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.finfo

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.Device

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
