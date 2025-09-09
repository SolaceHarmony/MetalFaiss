Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (init.md):
- Clear overview of nn.init package with examples; Sphinx table follows.
- Useful additions: guidance on when to prefer variants (He/Xavier) and dtype considerations.
-->

## Curated Notes

- Pick initializers based on activation: He/Kaiming for ReLU‑like, Xavier/Glorot for tanh/sigmoid; see `nn.init.he_uniform`, `he_normal`, `glorot_uniform`, `glorot_normal`.
- Ensure parameter dtype matches your training precision; cast after init if needed (e.g., to `float16`).
- Re‑initializing a model: `model.apply(init_fn)` applies to all modules that accept the initializer.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/init.rst"
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

# Initializers

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="initializers" class="section">

<span id="init"></span>

# Initializers<a href="https://ml-explore.github.io/mlx/build/html/#initializers"
class="headerlink" title="Link to this heading">#</a>

The <span class="pre">`mlx.nn.init`</span> package contains commonly
used initializers for neural network parameters. Initializers return a
function which can be applied to any input <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> to produce an
initialized output.

For example:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn

    init_fn = nn.init.uniform()

    # Produces a [2, 2] uniform matrix
    param = init_fn(mx.zeros((2, 2)))

</div>

</div>

To re-initialize all the parameter in an <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">mlx.nn.Module</code></span></a> from say a
uniform distribution, you can do:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.nn as nn
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 5))
    init_fn = nn.init.uniform(low=-0.1, high=0.1)
    model.apply(init_fn)

</div>

</div>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.constant.html#mlx.nn.init.constant"
class="reference internal" title="mlx.nn.init.constant"><span
class="pre"><code class="sourceCode python">constant</code></span></a>(value\[, dtype\]) | An initializer that returns an array filled with <span class="pre">`value`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.normal.html#mlx.nn.init.normal"
class="reference internal" title="mlx.nn.init.normal"><span
class="pre"><code class="sourceCode python">normal</code></span></a>(\[mean, std, dtype\]) | An initializer that returns samples from a normal distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.uniform.html#mlx.nn.init.uniform"
class="reference internal" title="mlx.nn.init.uniform"><span
class="pre"><code class="sourceCode python">uniform</code></span></a>(\[low, high, dtype\]) | An initializer that returns samples from a uniform distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.identity.html#mlx.nn.init.identity"
class="reference internal" title="mlx.nn.init.identity"><span
class="pre"><code class="sourceCode python">identity</code></span></a>(\[dtype\]) | An initializer that returns an identity matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_normal.html#mlx.nn.init.glorot_normal"
class="reference internal" title="mlx.nn.init.glorot_normal"><span
class="pre"><code
class="sourceCode python">glorot_normal</code></span></a>(\[dtype\]) | A Glorot normal initializer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_uniform.html#mlx.nn.init.glorot_uniform"
class="reference internal" title="mlx.nn.init.glorot_uniform"><span
class="pre"><code
class="sourceCode python">glorot_uniform</code></span></a>(\[dtype\]) | A Glorot uniform initializer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_normal.html#mlx.nn.init.he_normal"
class="reference internal" title="mlx.nn.init.he_normal"><span
class="pre"><code class="sourceCode python">he_normal</code></span></a>(\[dtype\]) | Build a He normal initializer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_uniform.html#mlx.nn.init.he_uniform"
class="reference internal" title="mlx.nn.init.he_uniform"><span
class="pre"><code class="sourceCode python">he_uniform</code></span></a>(\[dtype\]) | A He uniform (Kaiming uniform) initializer. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.losses.triplet_loss

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.constant.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.init.constant

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
