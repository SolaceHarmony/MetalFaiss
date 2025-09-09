Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers.rst"
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

# Optimizers

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#saving-and-loading"
  class="reference internal nav-link">Saving and Loading</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="optimizers" class="section">

<span id="id1"></span>

# Optimizers<a href="https://ml-explore.github.io/mlx/build/html/#optimizers"
class="headerlink" title="Link to this heading">#</a>

The optimizers in MLX can be used both with
<span class="pre">`mlx.nn`</span> but also with pure
<span class="pre">`mlx.core`</span> functions. A typical example
involves calling <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html#mlx.optimizers.Optimizer.update"
class="reference internal" title="mlx.optimizers.Optimizer.update"><span
class="pre"><code
class="sourceCode python">Optimizer.update()</code></span></a> to update
a model’s parameters based on the loss gradients and subsequently
calling <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python">mlx.core.<span class="bu">eval</span>()</code></span></a>
to evaluate both the model’s parameters and the **optimizer state**.

<div class="highlight-python notranslate">

<div class="highlight">

    # Create a model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    # Create the gradient function and the optimizer
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    for e in range(num_epochs):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)

            # Update the model with the gradients. So far no computation has happened.
            optimizer.update(model, grads)

            # Compute the new parameters but also the optimizer state.
            mx.eval(model.parameters(), optimizer.state)

</div>

</div>

<div id="saving-and-loading" class="section">

## Saving and Loading<a
href="https://ml-explore.github.io/mlx/build/html/#saving-and-loading"
class="headerlink" title="Link to this heading">#</a>

To serialize an optimizer, save its state. To load an optimizer, load
and set the saved state. Here’s a simple example:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    import mlx.optimizers as optim

    optimizer = optim.Adam(learning_rate=1e-2)

    # Perform some updates with the optimizer
    model = {"w" : mx.zeros((5, 5))}
    grads = {"w" : mx.ones((5, 5))}
    optimizer.update(model, grads)

    # Save the state
    state = tree_flatten(optimizer.state)
    mx.save_safetensors("optimizer.safetensors", dict(state))

    # Later on, for example when loading from a checkpoint,
    # recreate the optimizer and load the state
    optimizer = optim.Adam(learning_rate=1e-2)

    state = tree_unflatten(list(mx.load("optimizer.safetensors").items()))
    optimizer.state = state

</div>

</div>

Note, not every optimizer configuation parameter is saved in the state.
For example, for Adam the learning rate is saved but the
<span class="pre">`betas`</span> and <span class="pre">`eps`</span>
parameters are not. A good rule of thumb is if the parameter can be
scheduled then it will be included in the optimizer state.

<div class="toctree-wrapper compound">

- <a
  href="https://ml-explore.github.io/mlx/build/html/python/optimizers/optimizer.html"
  class="reference internal">Optimizer</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/optimizer.html#mlx.optimizers.Optimizer"
    class="reference internal"><span class="pre"><code
    class="docutils literal notranslate">Optimizer</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.state.html"
    class="reference internal">mlx.optimizers.Optimizer.state</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.state.html#mlx.optimizers.Optimizer.state"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Optimizer.state</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.apply_gradients.html"
    class="reference internal">mlx.optimizers.Optimizer.apply_gradients</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.apply_gradients.html#mlx.optimizers.Optimizer.apply_gradients"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Optimizer.apply_gradients()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.init.html"
    class="reference internal">mlx.optimizers.Optimizer.init</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.init.html#mlx.optimizers.Optimizer.init"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Optimizer.init()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html"
    class="reference internal">mlx.optimizers.Optimizer.update</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html#mlx.optimizers.Optimizer.update"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Optimizer.update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/optimizers/common_optimizers.html"
  class="reference internal">Common Optimizers</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.SGD.html"
    class="reference internal">mlx.optimizers.SGD</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.SGD.html#mlx.optimizers.SGD"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">SGD</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.RMSprop.html"
    class="reference internal">mlx.optimizers.RMSprop</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.RMSprop.html#mlx.optimizers.RMSprop"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">RMSprop</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adagrad.html"
    class="reference internal">mlx.optimizers.Adagrad</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adagrad.html#mlx.optimizers.Adagrad"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Adagrad</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adafactor.html"
    class="reference internal">mlx.optimizers.Adafactor</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adafactor.html#mlx.optimizers.Adafactor"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Adafactor</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdaDelta.html"
    class="reference internal">mlx.optimizers.AdaDelta</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdaDelta.html#mlx.optimizers.AdaDelta"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">AdaDelta</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adam.html"
    class="reference internal">mlx.optimizers.Adam</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adam.html#mlx.optimizers.Adam"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Adam</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html"
    class="reference internal">mlx.optimizers.AdamW</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html#mlx.optimizers.AdamW"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">AdamW</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adamax.html"
    class="reference internal">mlx.optimizers.Adamax</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adamax.html#mlx.optimizers.Adamax"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Adamax</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Lion.html"
    class="reference internal">mlx.optimizers.Lion</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Lion.html#mlx.optimizers.Lion"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Lion</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.MultiOptimizer.html"
    class="reference internal">mlx.optimizers.MultiOptimizer</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.MultiOptimizer.html#mlx.optimizers.MultiOptimizer"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">MultiOptimizer</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/optimizers/schedulers.html"
  class="reference internal">Schedulers</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.cosine_decay.html"
    class="reference internal">mlx.optimizers.cosine_decay</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.cosine_decay.html#mlx.optimizers.cosine_decay"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">cosine_decay()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.exponential_decay.html"
    class="reference internal">mlx.optimizers.exponential_decay</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.exponential_decay.html#mlx.optimizers.exponential_decay"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">exponential_decay()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.join_schedules.html"
    class="reference internal">mlx.optimizers.join_schedules</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.join_schedules.html#mlx.optimizers.join_schedules"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">join_schedules()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.linear_schedule.html"
    class="reference internal">mlx.optimizers.linear_schedule</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.linear_schedule.html#mlx.optimizers.linear_schedule"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">linear_schedule()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.step_decay.html"
    class="reference internal">mlx.optimizers.step_decay</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.step_decay.html#mlx.optimizers.step_decay"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">step_decay()</code></span></a>

</div>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.optimizers.clip_grad_norm.html#mlx.optimizers.clip_grad_norm"
class="reference internal" title="mlx.optimizers.clip_grad_norm"><span
class="pre"><code
class="sourceCode python">clip_grad_norm</code></span></a>(grads, max_norm) | Clips the global norm of the gradients. |

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_uniform.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.init.he_uniform

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/optimizer.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Optimizer

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#saving-and-loading"
  class="reference internal nav-link">Saving and Loading</a>

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
