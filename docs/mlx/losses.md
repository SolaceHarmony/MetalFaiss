Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (losses.md):
- Index of loss functions and signatures.
- Add notes on targets format and reduction semantics.
-->

## Curated Notes

- For classification: `cross_entropy(logits, targets)` expects integer class indices by default; ensure `axis` matches your class dimension.
- Many losses accept `reduction`: use `'none'` to inspect per‑example terms during debugging, then switch to `'mean'`/`'sum'` for training.
- For NLL/CE, make sure logits are not passed through softmax unless the API specifically wants probabilities.
- Precision tip: keep accumulations in `float32` for performance, but for very sensitive metrics you can compute reductions on CPU in `float64` and cast back for the rest of the pipeline.

### Examples

```python
import mlx.core as mx
import mlx.nn as nn

# Cross entropy with integer class targets
logits = mx.random.normal((8, 10))
targets = mx.random.randint(0, 10, (8,))
ce = nn.losses.cross_entropy(logits, targets)

# Binary cross entropy (probabilities)
probs = mx.sigmoid(mx.random.normal((8, 1)))
labels = mx.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [0.0], [1.0]])
bce = nn.losses.binary_cross_entropy(probs, labels)

# Inspect per-example terms
per = nn.losses.mse_loss(mx.zeros((4,)), mx.array([0.0, 1.0, 2.0, 3.0]), reduction="none")
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/losses.rst"
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

# Loss Functions

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="loss-functions" class="section">

<span id="losses"></span>

# Loss Functions<a href="https://ml-explore.github.io/mlx/build/html/#loss-functions"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.binary_cross_entropy.html#mlx.nn.losses.binary_cross_entropy"
class="reference internal"
title="mlx.nn.losses.binary_cross_entropy"><span class="pre"><code
class="sourceCode python">binary_cross_entropy</code></span></a>(inputs, targets\[, ...\]) | Computes the binary cross entropy loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cosine_similarity_loss.html#mlx.nn.losses.cosine_similarity_loss"
class="reference internal"
title="mlx.nn.losses.cosine_similarity_loss"><span class="pre"><code
class="sourceCode python">cosine_similarity_loss</code></span></a>(x1, x2\[, axis, eps, ...\]) | Computes the cosine similarity between the two inputs. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html#mlx.nn.losses.cross_entropy"
class="reference internal" title="mlx.nn.losses.cross_entropy"><span
class="pre"><code
class="sourceCode python">cross_entropy</code></span></a>(logits, targets\[, weights, ...\]) | Computes the cross entropy loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html#mlx.nn.losses.gaussian_nll_loss"
class="reference internal" title="mlx.nn.losses.gaussian_nll_loss"><span
class="pre"><code
class="sourceCode python">gaussian_nll_loss</code></span></a>(inputs, targets, vars\[, ...\]) | Computes the negative log likelihood loss for a Gaussian distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.hinge_loss.html#mlx.nn.losses.hinge_loss"
class="reference internal" title="mlx.nn.losses.hinge_loss"><span
class="pre"><code class="sourceCode python">hinge_loss</code></span></a>(inputs, targets\[, reduction\]) | Computes the hinge loss between inputs and targets. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.huber_loss.html#mlx.nn.losses.huber_loss"
class="reference internal" title="mlx.nn.losses.huber_loss"><span
class="pre"><code class="sourceCode python">huber_loss</code></span></a>(inputs, targets\[, delta, reduction\]) | Computes the Huber loss between inputs and targets. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.kl_div_loss.html#mlx.nn.losses.kl_div_loss"
class="reference internal" title="mlx.nn.losses.kl_div_loss"><span
class="pre"><code
class="sourceCode python">kl_div_loss</code></span></a>(inputs, targets\[, axis, reduction\]) | Computes the Kullback-Leibler divergence loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.l1_loss.html#mlx.nn.losses.l1_loss"
class="reference internal" title="mlx.nn.losses.l1_loss"><span
class="pre"><code class="sourceCode python">l1_loss</code></span></a>(predictions, targets\[, reduction\]) | Computes the L1 loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.log_cosh_loss.html#mlx.nn.losses.log_cosh_loss"
class="reference internal" title="mlx.nn.losses.log_cosh_loss"><span
class="pre"><code
class="sourceCode python">log_cosh_loss</code></span></a>(inputs, targets\[, reduction\]) | Computes the log cosh loss between inputs and targets. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.margin_ranking_loss.html#mlx.nn.losses.margin_ranking_loss"
class="reference internal"
title="mlx.nn.losses.margin_ranking_loss"><span class="pre"><code
class="sourceCode python">margin_ranking_loss</code></span></a>(inputs1, inputs2, targets) | Calculate the margin ranking loss that loss given inputs <span class="math notranslate nohighlight">\\x_1\\</span>, <span class="math notranslate nohighlight">\\x_2\\</span> and a label <span class="math notranslate nohighlight">\\y\\</span> (containing 1 or -1). |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.mse_loss.html#mlx.nn.losses.mse_loss"
class="reference internal" title="mlx.nn.losses.mse_loss"><span
class="pre"><code class="sourceCode python">mse_loss</code></span></a>(predictions, targets\[, reduction\]) | Computes the mean squared error loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.nll_loss.html#mlx.nn.losses.nll_loss"
class="reference internal" title="mlx.nn.losses.nll_loss"><span
class="pre"><code class="sourceCode python">nll_loss</code></span></a>(inputs, targets\[, axis, reduction\]) | Computes the negative log likelihood loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.smooth_l1_loss.html#mlx.nn.losses.smooth_l1_loss"
class="reference internal" title="mlx.nn.losses.smooth_l1_loss"><span
class="pre"><code
class="sourceCode python">smooth_l1_loss</code></span></a>(predictions, targets\[, beta, ...\]) | Computes the smooth L1 loss. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html#mlx.nn.losses.triplet_loss"
class="reference internal" title="mlx.nn.losses.triplet_loss"><span
class="pre"><code
class="sourceCode python">triplet_loss</code></span></a>(anchors, positives, negatives) | Computes the triplet loss for a set of anchor, positive, and negative samples. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.tanh.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.tanh

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.binary_cross_entropy.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.losses.binary_cross_entropy

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
