Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (linear_regression.md):
- Tutorial example with vanilla SGD using mx.grad; good for first-contact.
- Add variant using value_and_grad and discuss convergence checks.
-->

## Curated Variant (value_and_grad)

```python
opt_w = mx.random.normal((num_features,))

def loss_w(w):
    return 0.5 * mx.mean((X @ w - y) ** 2)

for step in range(2000):
    loss, grad = mx.value_and_grad(loss_w)(opt_w)
    opt_w = opt_w - lr * grad
    if step % 200 == 0:
        print(step, loss.item())
```

Tip: monitor a validation split or a held‑out metric to avoid overfitting when you extend this to NN models.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/examples/linear_regression.rst"
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

# Linear Regression

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="linear-regression" class="section">

<span id="id1"></span>

# Linear Regression<a href="https://ml-explore.github.io/mlx/build/html/#linear-regression"
class="headerlink" title="Link to this heading">#</a>

Let’s implement a basic linear regression model as a starting point to
learn MLX. First import the core package and setup some problem
metadata:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx

    num_features = 100
    num_examples = 1_000
    num_iters = 10_000  # iterations of SGD
    lr = 0.01  # learning rate for SGD

</div>

</div>

We’ll generate a synthetic dataset by:

1.  Sampling the design matrix <span class="pre">`X`</span>.

2.  Sampling a ground truth parameter vector
    <span class="pre">`w_star`</span>.

3.  Compute the dependent values <span class="pre">`y`</span> by adding
    Gaussian noise to
    <span class="pre">`X`</span>` `<span class="pre">`@`</span>` `<span class="pre">`w_star`</span>.

<div class="highlight-python notranslate">

<div class="highlight">

    # True parameters
    w_star = mx.random.normal((num_features,))

    # Input examples (design matrix)
    X = mx.random.normal((num_examples, num_features))

    # Noisy labels
    eps = 1e-2 * mx.random.normal((num_examples,))
    y = X @ w_star + eps

</div>

</div>

We will use SGD to find the optimal weights. To start, define the
squared loss and get the gradient function of the loss with respect to
the parameters.

<div class="highlight-python notranslate">

<div class="highlight">

    def loss_fn(w):
        return 0.5 * mx.mean(mx.square(X @ w - y))

    grad_fn = mx.grad(loss_fn)

</div>

</div>

Start the optimization by initializing the parameters
<span class="pre">`w`</span> randomly. Then repeatedly update the
parameters for <span class="pre">`num_iters`</span> iterations.

<div class="highlight-python notranslate">

<div class="highlight">

    w = 1e-2 * mx.random.normal((num_features,))

    for _ in range(num_iters):
        grad = grad_fn(w)
        w = w - lr * grad
        mx.eval(w)

</div>

</div>

Finally, compute the loss of the learned parameters and verify that they
are close to the ground truth parameters.

<div class="highlight-python notranslate">

<div class="highlight">

    loss = loss_fn(w)
    error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5

    print(
        f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}, "
    )
    # Should print something close to: Loss 0.00005, |w-w*| = 0.00364

</div>

</div>

Complete <a
href="https://github.com/ml-explore/mlx/tree/main/examples/python/linear_regression.py"
class="reference external">linear regression</a> and <a
href="https://github.com/ml-explore/mlx/tree/main/examples/python/logistic_regression.py"
class="reference external">logistic regression</a> examples are
available in the MLX GitHub repo.

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/usage/export.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Exporting Functions

</div>

<a href="https://ml-explore.github.io/mlx/build/html/examples/mlp.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Multi-Layer Perceptron

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
