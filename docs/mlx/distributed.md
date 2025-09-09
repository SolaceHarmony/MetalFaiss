Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (distributed.md):
- API surface for distributed primitives via MPI; includes collectives and P2P.
- Users need concrete init guidance and shape/dtype alignment reminders.
-->

## Curated Notes

- Initialize once per process: `from mlx.core.distributed import init, is_available; init() if is_available() else ...`.
- Ensure both sender and receiver agree on `shape`/`dtype`; mismatches will fail at runtime.
- Prefer collectives (`all_sum`, `all_gather`) for data/grad sync; use `send/recv` for pipeline or sparse paths.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/distributed.rst"
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

# Distributed Communication

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="distributed-communication" class="section">

<span id="distributed"></span>

# Distributed Communication<a
href="https://ml-explore.github.io/mlx/build/html/#distributed-communication"
class="headerlink" title="Link to this heading">#</a>

MLX provides a distributed communication package using MPI. The MPI
library is loaded at runtime; if MPI is available then distributed
communication is also made available.

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.Group.html#mlx.core.distributed.Group"
class="reference internal" title="mlx.core.distributed.Group"><span
class="pre"><code class="sourceCode python">Group</code></span></a> | An <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.Group.html#mlx.core.distributed.Group"
class="reference internal" title="mlx.core.distributed.Group"><span
class="pre"><code
class="sourceCode python">mlx.core.distributed.Group</code></span></a> represents a group of independent mlx processes that can communicate. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.is_available.html#mlx.core.distributed.is_available"
class="reference internal"
title="mlx.core.distributed.is_available"><span class="pre"><code
class="sourceCode python">is_available</code></span></a>() | Check if a communication backend is available. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.init.html#mlx.core.distributed.init"
class="reference internal" title="mlx.core.distributed.init"><span
class="pre"><code class="sourceCode python">init</code></span></a>(\[strict, backend\]) | Initialize the communication backend and create the global communication group. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.all_sum.html#mlx.core.distributed.all_sum"
class="reference internal" title="mlx.core.distributed.all_sum"><span
class="pre"><code class="sourceCode python">all_sum</code></span></a>(x, \*\[, group, stream\]) | All reduce sum. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.all_gather.html#mlx.core.distributed.all_gather"
class="reference internal" title="mlx.core.distributed.all_gather"><span
class="pre"><code class="sourceCode python">all_gather</code></span></a>(x, \*\[, group, stream\]) | Gather arrays from all processes. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.send.html#mlx.core.distributed.send"
class="reference internal" title="mlx.core.distributed.send"><span
class="pre"><code class="sourceCode python">send</code></span></a>(x, dst, \*\[, group, stream\]) | Send an array from the current process to the process that has rank <span class="pre">`dst`</span> in the group. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.recv.html#mlx.core.distributed.recv"
class="reference internal" title="mlx.core.distributed.recv"><span
class="pre"><code class="sourceCode python">recv</code></span></a>(shape, dtype, src, \*\[, group, stream\]) | Recv an array with shape <span class="pre">`shape`</span> and dtype <span class="pre">`dtype`</span> from process with rank <span class="pre">`src`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.recv_like.html#mlx.core.distributed.recv_like"
class="reference internal" title="mlx.core.distributed.recv_like"><span
class="pre"><code class="sourceCode python">recv_like</code></span></a>(x, src, \*\[, group, stream\]) | Recv an array with shape and type like <span class="pre">`x`</span> from process with rank <span class="pre">`src`</span>. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.optimizers.clip_grad_norm.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.clip_grad_norm

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.Group.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.distributed.Group

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
