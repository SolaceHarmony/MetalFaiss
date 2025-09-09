Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (mlp.md):
- MNIST MLP example using nn.Module and manual loop.
- Add curated training loop with optimizer and evaluation toggle.
-->

## Curated Notes

```python
from mlx.optimizers import AdamW

model = MLP(num_layers=2, input_dim=784, hidden_dim=256, output_dim=10)
params = model.parameters()
opt = AdamW(3e-4)

def loss_fn(p, x, y):
    logits = model.apply(p, x)
    return nn.losses.cross_entropy(logits, y)

for step, (xb, yb) in enumerate(train_loader):
    loss, grads = mx.value_and_grad(loss_fn)(params, xb, yb)
    params = opt.update(params, grads)
    if step % 100 == 0:
        print(step, loss.item())

model.eval()
```

Switch to `model.eval()` for validation to disable dropout and training‑mode behavior in norms.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/examples/mlp.rst"
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

# Multi-Layer Perceptron

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="multi-layer-perceptron" class="section">

<span id="mlp"></span>

# Multi-Layer Perceptron<a
href="https://ml-explore.github.io/mlx/build/html/#multi-layer-perceptron"
class="headerlink" title="Link to this heading">#</a>

In this example we’ll learn to use <span class="pre">`mlx.nn`</span> by
implementing a simple multi-layer perceptron to classify MNIST.

As a first step import the MLX packages we need:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    import numpy as np

</div>

</div>

The model is defined as the <span class="pre">`MLP`</span> class which
inherits from <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">mlx.nn.Module</code></span></a>. We follow the
standard idiom to make a new module:

1.  Define an <span class="pre">`__init__`</span> where the parameters
    and/or submodules are setup. See the <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn.html#module-class"
    class="reference internal"><span class="std std-ref">Module class
    docs</span></a> for more information on how <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
    class="reference internal" title="mlx.nn.Module"><span class="pre"><code
    class="sourceCode python">mlx.nn.Module</code></span></a> registers
    parameters.

2.  Define a <span class="pre">`__call__`</span> where the computation
    is implemented.

<div class="highlight-python notranslate">

<div class="highlight">

    class MLP(nn.Module):
        def __init__(
            self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
        ):
            super().__init__()
            layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
            self.layers = [
                nn.Linear(idim, odim)
                for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
            ]

        def __call__(self, x):
            for l in self.layers[:-1]:
                x = mx.maximum(l(x), 0.0)
            return self.layers[-1](x)

</div>

</div>

We define the loss function which takes the mean of the per-example
cross entropy loss. The <span class="pre">`mlx.nn.losses`</span>
sub-package has implementations of some commonly used loss functions.

<div class="highlight-python notranslate">

<div class="highlight">

    def loss_fn(model, X, y):
        return mx.mean(nn.losses.cross_entropy(model(X), y))

</div>

</div>

We also need a function to compute the accuracy of the model on the
validation set:

<div class="highlight-python notranslate">

<div class="highlight">

    def eval_fn(model, X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

</div>

</div>

Next, setup the problem parameters and load the data. To load the data,
you need our <a
href="https://github.com/ml-explore/mlx-examples/blob/main/mnist/mnist.py"
class="reference external">mnist data loader</a>, which we will import
as <span class="pre">`mnist`</span>.

<div class="highlight-python notranslate">

<div class="highlight">

    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    # Load the data
    import mnist
    train_images, train_labels, test_images, test_labels = map(
        mx.array, mnist.mnist()
    )

</div>

</div>

Since we’re using SGD, we need an iterator which shuffles and constructs
minibatches of examples in the training set:

<div class="highlight-python notranslate">

<div class="highlight">

    def batch_iterate(batch_size, X, y):
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, batch_size):
            ids = perm[s : s + batch_size]
            yield X[ids], y[ids]

</div>

</div>

Finally, we put it all together by instantiating the model, the <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.SGD.html#mlx.optimizers.SGD"
class="reference internal" title="mlx.optimizers.SGD"><span
class="pre"><code
class="sourceCode python">mlx.optimizers.SGD</code></span></a>
optimizer, and running the training loop:

<div class="highlight-python notranslate">

<div class="highlight">

    # Load the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    # Get a function which gives the loss and gradient of the
    # loss with respect to the model's trainable parameters
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Instantiate the optimizer
    optimizer = optim.SGD(learning_rate=learning_rate)

    for e in range(num_epochs):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)

            # Update the optimizer state and model parameters
            # in a single call
            optimizer.update(model, grads)

            # Force a graph evaluation
            mx.eval(model.parameters(), optimizer.state)

        accuracy = eval_fn(model, test_images, test_labels)
        print(f"Epoch {e}: Test accuracy {accuracy.item():.3f}")

</div>

</div>

<div class="admonition note">

Note

The <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
class="reference internal" title="mlx.nn.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.nn.value_and_grad()</code></span></a>
function is a convenience function to get the gradient of a loss with
respect to the trainable parameters of a model. This should not be
confused with <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.core.value_and_grad()</code></span></a>.

</div>

The model should train to a decent accuracy (about 95%) after just a few
passes over the training set. The
<a href="https://github.com/ml-explore/mlx-examples/tree/main/mnist"
class="reference external">full example</a> is available in the MLX
GitHub repo.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/examples/linear_regression.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Linear Regression

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

LLM inference

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
