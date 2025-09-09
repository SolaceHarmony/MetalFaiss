Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (metal_debugger.md):
- How to enable Metal debugging and capture traces; good step-by-step.
- Add curated reminders about non-existent path requirement and capture scoping.
-->

## Curated Notes

- Ensure `trace_file` does not exist before capture; some systems won’t overwrite and will silently fail to start capture.
- Scope captures tightly to the region you’re profiling to keep traces small and readable.
- Label custom kernels and streams where possible to improve Xcode timeline clarity.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/dev/metal_debugger.rst"
  class="btn btn-sm btn-download-source-button dropdown-item"
  data-bs-placement="left" data-bs-toggle="tooltip" target="_blank"
  title="Download source file"><span class="btn__icon-container">
  <em></em> </span> <span class="btn__text-container">.rst</span></a>
- <span class="btn__icon-container"> </span>
  <span class="btn__text-container">.pdf</span>

</div>

<span class="btn__icon-container"> </span>

<span class="fa-solid fa-list"></span>

</div>

</div>

</div>

</div>

</div>

<div id="jb-print-docs-body" class="onlyprint">

# Metal Debugger

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#xcode-workflow"
  class="reference internal nav-link">Xcode Workflow</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="metal-debugger" class="section">

# Metal Debugger<a href="https://ml-explore.github.io/mlx/build/html/#metal-debugger"
class="headerlink" title="Link to this heading">#</a>

Profiling is a key step for performance optimization. You can build MLX
with the <span class="pre">`MLX_METAL_DEBUG`</span> option to improve
the Metal debugging and optimization workflow. The
<span class="pre">`MLX_METAL_DEBUG`</span> debug option:

- Records source during Metal compilation, for later inspection while
  debugging.

- Labels Metal objects such as command queues, improving capture
  readability.

To build with debugging enabled in Python prepend
<span class="pre">`CMAKE_ARGS="-DMLX_METAL_DEBUG=ON"`</span> to the
build call.

The <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.start_capture.html#mlx.core.metal.start_capture"
class="reference internal" title="mlx.core.metal.start_capture"><span
class="pre"><code
class="sourceCode python">metal.start_capture()</code></span></a>
function initiates a capture of all MLX GPU work.

<div class="admonition note">

Note

To capture a GPU trace you must run the application with
<span class="pre">`MTL_CAPTURE_ENABLED=1`</span>.

</div>

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx

    a = mx.random.uniform(shape=(512, 512))
    b = mx.random.uniform(shape=(512, 512))
    mx.eval(a, b)

    trace_file = "mlx_trace.gputrace"

    # Make sure to run with MTL_CAPTURE_ENABLED=1 and
    # that the path trace_file does not already exist.
    mx.metal.start_capture(trace_file)

    for _ in range(10):
      mx.eval(mx.add(a, b))

    mx.metal.stop_capture()

</div>

</div>

You can open and replay the GPU trace in Xcode. The
<span class="pre">`Dependencies`</span> view has a great overview of all
operations. Checkout the
<a href="https://developer.apple.com/documentation/xcode/metal-debugger"
class="reference external">Metal debugger documentation</a> for more
information.

<img
src="https://ml-explore.github.io/mlx/build/html/_images/capture.png"
class="dark-light" alt="../_images/capture.png" />

<div id="xcode-workflow" class="section">

## Xcode Workflow<a href="https://ml-explore.github.io/mlx/build/html/#xcode-workflow"
class="headerlink" title="Link to this heading">#</a>

You can skip saving to a path by running within Xcode. First, generate
an Xcode project using CMake.

<div class="highlight-python notranslate">

<div class="highlight">

    mkdir build && cd build
    cmake .. -DMLX_METAL_DEBUG=ON -G Xcode
    open mlx.xcodeproj

</div>

</div>

Select the <span class="pre">`metal_capture`</span> example schema and
run.

<img
src="https://ml-explore.github.io/mlx/build/html/_images/schema.png"
class="dark-light" alt="../_images/schema.png" />

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/dev/extensions.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Custom Extensions in MLX

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Custom Metal Kernels

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#xcode-workflow"
  class="reference internal nav-link">Xcode Workflow</a>

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
