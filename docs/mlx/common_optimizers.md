Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (common_optimizers.md):
- Sphinx table of optimizer classes; heavy HTML wrappers present.
- Common real-world confusion: clip_grad_norm return type and schedule wiring.
-->

## Usage Pattern (curated)

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW, clip_grad_norm, cosine_decay

net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 10))
params = net.parameters()
opt = AdamW(cosine_decay(3e-4, total_steps=10000))

def loss_fn(p, x, y):
    logits = net.apply(p, x)
    return mx.mean((logits - y) ** 2)

x = mx.random.normal((128, 32)); y = mx.random.normal((128, 10))
loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
grads, _ = clip_grad_norm(grads, 1.0)   # unpack tuple
params = opt.update(params, grads)
```

### More Examples

```python
# SGD with momentum and weight decay
from mlx.optimizers import SGD
opt = SGD(1e-2, momentum=0.9, weight_decay=1e-4)

# Freezing a subtree (e.g., encoder) by zeroing grads
from mlx.utils import tree_map_with_path
def zero_frozen(path, g):
    return mx.zeros_like(g) if path and path[0] == "encoder" else g
loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
grads = tree_map_with_path(zero_frozen, grads)
params = opt.update(params, grads)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/common_optimizers.rst"
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

# Common Optimizers

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="common-optimizers" class="section">

<span id="id1"></span>

# Common Optimizers<a href="https://ml-explore.github.io/mlx/build/html/#common-optimizers"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.SGD.html#mlx.optimizers.SGD"
class="reference internal" title="mlx.optimizers.SGD"><span
class="pre"><code class="sourceCode python">SGD</code></span></a>(learning_rate\[, momentum, weight_decay, ...\]) | The stochastic gradient descent optimizer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.RMSprop.html#mlx.optimizers.RMSprop"
class="reference internal" title="mlx.optimizers.RMSprop"><span
class="pre"><code class="sourceCode python">RMSprop</code></span></a>(learning_rate\[, alpha, eps\]) | The RMSprop optimizer \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adagrad.html#mlx.optimizers.Adagrad"
class="reference internal" title="mlx.optimizers.Adagrad"><span
class="pre"><code class="sourceCode python">Adagrad</code></span></a>(learning_rate\[, eps\]) | The Adagrad optimizer \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adafactor.html#mlx.optimizers.Adafactor"
class="reference internal" title="mlx.optimizers.Adafactor"><span
class="pre"><code class="sourceCode python">Adafactor</code></span></a>(\[learning_rate, eps, ...\]) | The Adafactor optimizer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdaDelta.html#mlx.optimizers.AdaDelta"
class="reference internal" title="mlx.optimizers.AdaDelta"><span
class="pre"><code class="sourceCode python">AdaDelta</code></span></a>(learning_rate\[, rho, eps\]) | The AdaDelta optimizer with a learning rate \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adam.html#mlx.optimizers.Adam"
class="reference internal" title="mlx.optimizers.Adam"><span
class="pre"><code class="sourceCode python">Adam</code></span></a>(learning_rate\[, betas, eps, ...\]) | The Adam optimizer \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html#mlx.optimizers.AdamW"
class="reference internal" title="mlx.optimizers.AdamW"><span
class="pre"><code class="sourceCode python">AdamW</code></span></a>(learning_rate\[, betas, eps, ...\]) | The AdamW optimizer \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adamax.html#mlx.optimizers.Adamax"
class="reference internal" title="mlx.optimizers.Adamax"><span
class="pre"><code class="sourceCode python">Adamax</code></span></a>(learning_rate\[, betas, eps\]) | The Adamax optimizer, a variant of Adam based on the infinity norm \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Lion.html#mlx.optimizers.Lion"
class="reference internal" title="mlx.optimizers.Lion"><span
class="pre"><code class="sourceCode python">Lion</code></span></a>(learning_rate\[, betas, weight_decay\]) | The Lion optimizer \[1\]. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.MultiOptimizer.html#mlx.optimizers.MultiOptimizer"
class="reference internal" title="mlx.optimizers.MultiOptimizer"><span
class="pre"><code
class="sourceCode python">MultiOptimizer</code></span></a>(optimizers\[, filters\]) | Wraps a list of optimizers with corresponding weight predicates/filters to make it easy to use different optimizers for different weights. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Optimizer.update

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.SGD.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.SGD

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
