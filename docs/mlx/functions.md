Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (functions.md):
- Index of stateless NN functions (activations, GLU, etc.).
- Value: remind shape/axis expectations and numerics for fast approximations.
-->

## Curated Notes

- Many activations have both module and function forms; choose by preference (no parameters -> function is convenient).
- `glu(x, axis=...)`: validate that the chosen `axis` can be split evenly into two gates.
- Approximate variants (e.g., `gelu_approx`) trade accuracy for speed; verify on your domain.

### Examples

```python
import mlx.core as mx
import mlx.nn as nn

x = mx.random.normal((2, 6))
print(nn.relu(x))

# GLU splits features along axis and gates half by sigmoid(other half)
g = nn.glu(x, axis=-1)   # requires last dim even (splits into 3+3)
print(g.shape)

# Function vs Module forms
relu_mod = nn.ReLU()
print(relu_mod(x))
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/functions.rst"
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

# Functions

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="functions" class="section">

<span id="nn-functions"></span>

# Functions<a href="https://ml-explore.github.io/mlx/build/html/#functions"
class="headerlink" title="Link to this heading">#</a>

Layers without parameters (e.g. activation functions) are also provided
as simple functions.

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.elu.html#mlx.nn.elu"
class="reference internal" title="mlx.nn.elu"><span class="pre"><code
class="sourceCode python">elu</code></span></a> | elu(x, alpha=1.0) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.celu.html#mlx.nn.celu"
class="reference internal" title="mlx.nn.celu"><span class="pre"><code
class="sourceCode python">celu</code></span></a> | celu(x, alpha=1.0) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu.html#mlx.nn.gelu"
class="reference internal" title="mlx.nn.gelu"><span class="pre"><code
class="sourceCode python">gelu</code></span></a> | gelu(x) -\> mlx.core.array |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html#mlx.nn.gelu_approx"
class="reference internal" title="mlx.nn.gelu_approx"><span
class="pre"><code
class="sourceCode python">gelu_approx</code></span></a> | gelu_approx(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_fast_approx.html#mlx.nn.gelu_fast_approx"
class="reference internal" title="mlx.nn.gelu_fast_approx"><span
class="pre"><code
class="sourceCode python">gelu_fast_approx</code></span></a> | gelu_fast_approx(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.glu.html#mlx.nn.glu"
class="reference internal" title="mlx.nn.glu"><span class="pre"><code
class="sourceCode python">glu</code></span></a>(x\[, axis\]) | Applies the gated linear unit function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hard_shrink.html#mlx.nn.hard_shrink"
class="reference internal" title="mlx.nn.hard_shrink"><span
class="pre"><code
class="sourceCode python">hard_shrink</code></span></a> | hard_shrink(x, lambd=0.5) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hard_tanh.html#mlx.nn.hard_tanh"
class="reference internal" title="mlx.nn.hard_tanh"><span
class="pre"><code class="sourceCode python">hard_tanh</code></span></a> | hard_tanh(x, min_val=-1.0, max_val=1.0) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hardswish.html#mlx.nn.hardswish"
class="reference internal" title="mlx.nn.hardswish"><span
class="pre"><code class="sourceCode python">hardswish</code></span></a> | hardswish(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.leaky_relu.html#mlx.nn.leaky_relu"
class="reference internal" title="mlx.nn.leaky_relu"><span
class="pre"><code class="sourceCode python">leaky_relu</code></span></a> | leaky_relu(x, negative_slope=0.01) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.log_sigmoid.html#mlx.nn.log_sigmoid"
class="reference internal" title="mlx.nn.log_sigmoid"><span
class="pre"><code
class="sourceCode python">log_sigmoid</code></span></a> | log_sigmoid(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.log_softmax.html#mlx.nn.log_softmax"
class="reference internal" title="mlx.nn.log_softmax"><span
class="pre"><code
class="sourceCode python">log_softmax</code></span></a> | log_softmax(x, axis=-1) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.mish.html#mlx.nn.mish"
class="reference internal" title="mlx.nn.mish"><span class="pre"><code
class="sourceCode python">mish</code></span></a> | mlx.core.array) -\> mlx.core.array |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.prelu.html#mlx.nn.prelu"
class="reference internal" title="mlx.nn.prelu"><span class="pre"><code
class="sourceCode python">prelu</code></span></a> | mlx.core.array) -\> mlx.core.array |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu.html#mlx.nn.relu"
class="reference internal" title="mlx.nn.relu"><span class="pre"><code
class="sourceCode python">relu</code></span></a> | relu(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu6.html#mlx.nn.relu6"
class="reference internal" title="mlx.nn.relu6"><span class="pre"><code
class="sourceCode python">relu6</code></span></a> | relu6(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.selu.html#mlx.nn.selu"
class="reference internal" title="mlx.nn.selu"><span class="pre"><code
class="sourceCode python">selu</code></span></a> | selu(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.sigmoid.html#mlx.nn.sigmoid"
class="reference internal" title="mlx.nn.sigmoid"><span
class="pre"><code class="sourceCode python">sigmoid</code></span></a> | sigmoid(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.silu.html#mlx.nn.silu"
class="reference internal" title="mlx.nn.silu"><span class="pre"><code
class="sourceCode python">silu</code></span></a> | silu(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softmax.html#mlx.nn.softmax"
class="reference internal" title="mlx.nn.softmax"><span
class="pre"><code class="sourceCode python">softmax</code></span></a> | softmax(x, axis=-1) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softmin.html#mlx.nn.softmin"
class="reference internal" title="mlx.nn.softmin"><span
class="pre"><code class="sourceCode python">softmin</code></span></a> | softmin(x, axis=-1) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softplus.html#mlx.nn.softplus"
class="reference internal" title="mlx.nn.softplus"><span
class="pre"><code class="sourceCode python">softplus</code></span></a> | softplus(x) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softshrink.html#mlx.nn.softshrink"
class="reference internal" title="mlx.nn.softshrink"><span
class="pre"><code class="sourceCode python">softshrink</code></span></a> | float = 0.5) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.step.html#mlx.nn.step"
class="reference internal" title="mlx.nn.step"><span class="pre"><code
class="sourceCode python">step</code></span></a> | float = 0.0) |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.tanh.html#mlx.nn.tanh"
class="reference internal" title="mlx.nn.tanh"><span class="pre"><code
class="sourceCode python">tanh</code></span></a>(x) | Applies the hyperbolic tangent function. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Upsample.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Upsample

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.elu.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.elu

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
