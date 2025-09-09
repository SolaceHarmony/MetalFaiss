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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn.rst"
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

# Neural Networks

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#quick-start-with-neural-networks"
  class="reference internal nav-link">Quick Start with Neural Networks</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#the-module-class"
  class="reference internal nav-link">The Module Class</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#parameters"
    class="reference internal nav-link">Parameters</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#updating-the-parameters"
    class="reference internal nav-link">Updating the Parameters</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#inspecting-modules"
    class="reference internal nav-link">Inspecting Modules</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#value-and-grad"
  class="reference internal nav-link">Value and Grad</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="neural-networks" class="section">

<span id="nn"></span>

# Neural Networks<a href="https://ml-explore.github.io/mlx/build/html/#neural-networks"
class="headerlink" title="Link to this heading">#</a>

Writing arbitrarily complex neural networks in MLX can be done using
only <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.core.value_and_grad()</code></span></a>.
However, this requires the user to write again and again the same simple
neural network operations as well as handle all the parameter state and
initialization manually and explicitly.

The module <span class="pre">`mlx.nn`</span> solves this problem by
providing an intuitive way of composing neural network layers,
initializing their parameters, freezing them for finetuning and more.

<div id="quick-start-with-neural-networks" class="section">

## Quick Start with Neural Networks<a
href="https://ml-explore.github.io/mlx/build/html/#quick-start-with-neural-networks"
class="headerlink" title="Link to this heading">#</a>

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int):
            super().__init__()

            self.layers = [
                nn.Linear(in_dims, 128),
                nn.Linear(128, 128),
                nn.Linear(128, out_dims),
            ]

        def __call__(self, x):
            for i, l in enumerate(self.layers):
                x = mx.maximum(x, 0) if i > 0 else x
                x = l(x)
            return x

    # The model is created with all its parameters but nothing is initialized
    # yet because MLX is lazily evaluated
    mlp = MLP(2, 10)

    # We can access its parameters by calling mlp.parameters()
    params = mlp.parameters()
    print(params["layers"][0]["weight"].shape)

    # Printing a parameter will cause it to be evaluated and thus initialized
    print(params["layers"][0])

    # We can also force evaluate all parameters to initialize the model
    mx.eval(mlp.parameters())

    # A simple loss function.
    # NOTE: It doesn't matter how it uses the mlp model. It currently captures
    #       it from the local scope. It could be a positional argument or a
    #       keyword argument.
    def l2_loss(x, y):
        y_hat = mlp(x)
        return (y_hat - y).square().mean()

    # Calling `nn.value_and_grad` instead of `mx.value_and_grad` returns the
    # gradient with respect to `mlp.trainable_parameters()`
    loss_and_grad = nn.value_and_grad(mlp, l2_loss)

</div>

</div>

</div>

<div id="the-module-class" class="section">

<span id="module-class"></span>

## The Module Class<a href="https://ml-explore.github.io/mlx/build/html/#the-module-class"
class="headerlink" title="Link to this heading">#</a>

The workhorse of any neural network library is the <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> class. In MLX the <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> class is a container
of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> or <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> instances. Its main
function is to provide a way to recursively **access** and **update**
its parameters and those of its submodules.

<div id="parameters" class="section">

### Parameters<a href="https://ml-explore.github.io/mlx/build/html/#parameters"
class="headerlink" title="Link to this heading">#</a>

A parameter of a module is any public member of type <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> (its name
should not start with <span class="pre">`_`</span>). It can be
arbitrarily nested in other <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> instances or lists
and dictionaries.

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters"
class="reference internal" title="mlx.nn.Module.parameters"><span
class="pre"><code
class="sourceCode python">Module.parameters()</code></span></a> can be
used to extract a nested dictionary with all the parameters of a module
and its submodules.

A <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> can also keep track
of “frozen” parameters. See the <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.freeze.html#mlx.nn.Module.freeze"
class="reference internal" title="mlx.nn.Module.freeze"><span
class="pre"><code
class="sourceCode python">Module.freeze()</code></span></a> method for
more details. <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
class="reference internal" title="mlx.nn.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.nn.value_and_grad()</code></span></a> the
gradients returned will be with respect to these trainable parameters.

</div>

<div id="updating-the-parameters" class="section">

### Updating the Parameters<a
href="https://ml-explore.github.io/mlx/build/html/#updating-the-parameters"
class="headerlink" title="Link to this heading">#</a>

MLX modules allow accessing and updating individual parameters. However,
most times we need to update large subsets of a module’s parameters.
This action is performed by <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update"
class="reference internal" title="mlx.nn.Module.update"><span
class="pre"><code
class="sourceCode python">Module.update()</code></span></a>.

</div>

<div id="inspecting-modules" class="section">

### Inspecting Modules<a
href="https://ml-explore.github.io/mlx/build/html/#inspecting-modules"
class="headerlink" title="Link to this heading">#</a>

The simplest way to see the model architecture is to print it. Following
along with the above example, you can print the
<span class="pre">`MLP`</span> with:

<div class="highlight-python notranslate">

<div class="highlight">

    print(mlp)

</div>

</div>

This will display:

<div class="highlight-shell notranslate">

<div class="highlight">

    MLP(
      (layers.0): Linear(input_dims=2, output_dims=128, bias=True)
      (layers.1): Linear(input_dims=128, output_dims=128, bias=True)
      (layers.2): Linear(input_dims=128, output_dims=10, bias=True)
    )

</div>

</div>

To get more detailed information on the arrays in a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> you can use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map.html#mlx.utils.tree_map"
class="reference internal" title="mlx.utils.tree_map"><span
class="pre"><code
class="sourceCode python">mlx.utils.tree_map()</code></span></a> on the
parameters. For example, to see the shapes of all the parameters in a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> do:

<div class="highlight-python notranslate">

<div class="highlight">

    from mlx.utils import tree_map
    shapes = tree_map(lambda p: p.shape, mlp.parameters())

</div>

</div>

As another example, you can count the number of parameters in a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> with:

<div class="highlight-python notranslate">

<div class="highlight">

    from mlx.utils import tree_flatten
    num_params = sum(v.size for _, v in tree_flatten(mlp.parameters()))

</div>

</div>

</div>

</div>

<div id="value-and-grad" class="section">

## Value and Grad<a href="https://ml-explore.github.io/mlx/build/html/#value-and-grad"
class="headerlink" title="Link to this heading">#</a>

Using a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> does not preclude
using MLX’s high order function transformations (<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.core.value_and_grad()</code></span></a>,
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">mlx.core.grad()</code></span></a>, etc.).
However, these function transformations assume pure functions, namely
the parameters should be passed as an argument to the function being
transformed.

There is an easy pattern to achieve that with MLX modules

<div class="highlight-python notranslate">

<div class="highlight">

    model = ...

    def f(params, other_inputs):
        model.update(params)  # <---- Necessary to make the model use the passed parameters
        return model(other_inputs)

    f(model.trainable_parameters(), mx.zeros((10,)))

</div>

</div>

However, <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
class="reference internal" title="mlx.nn.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.nn.value_and_grad()</code></span></a>
provides precisely this pattern and only computes the gradients with
respect to the trainable parameters of the model.

In detail:

- it wraps the passed function with a function that calls <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update"
  class="reference internal" title="mlx.nn.Module.update"><span
  class="pre"><code
  class="sourceCode python">Module.update()</code></span></a> to make
  sure the model is using the provided parameters.

- it calls <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
  class="reference internal" title="mlx.core.value_and_grad"><span
  class="pre"><code
  class="sourceCode python">mlx.core.value_and_grad()</code></span></a>
  to transform the function into a function that also computes the
  gradients with respect to the passed parameters.

- it wraps the returned function with a function that passes the
  trainable parameters as the first argument to the function returned by
  <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
  class="reference internal" title="mlx.core.value_and_grad"><span
  class="pre"><code
  class="sourceCode python">mlx.core.value_and_grad()</code></span></a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
class="reference internal" title="mlx.nn.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad</code></span></a>(model, fn) | Transform the passed function <span class="pre">`fn`</span> to a function that computes the gradients of <span class="pre">`fn`</span> wrt the model's trainable parameters and also its value. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html#mlx.nn.quantize"
class="reference internal" title="mlx.nn.quantize"><span
class="pre"><code class="sourceCode python">quantize</code></span></a>(model\[, group_size, bits, ...\]) | Quantize the sub-modules of a module according to a predicate. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.average_gradients.html#mlx.nn.average_gradients"
class="reference internal" title="mlx.nn.average_gradients"><span
class="pre"><code
class="sourceCode python">average_gradients</code></span></a>(gradients\[, group, ...\]) | Average the gradients across the distributed processes in the passed group. |

</div>

<div class="toctree-wrapper compound">

- <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html"
  class="reference internal">Module</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
    class="reference internal"><span class="pre"><code
    class="docutils literal notranslate">Module</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.training.html"
    class="reference internal">mlx.nn.Module.training</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.training.html#mlx.nn.Module.training"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.training</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.state.html"
    class="reference internal">mlx.nn.Module.state</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.state.html#mlx.nn.Module.state"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.state</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.apply.html"
    class="reference internal">mlx.nn.Module.apply</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.apply.html#mlx.nn.Module.apply"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.apply()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.apply_to_modules.html"
    class="reference internal">mlx.nn.Module.apply_to_modules</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.apply_to_modules.html#mlx.nn.Module.apply_to_modules"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.apply_to_modules()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.children.html"
    class="reference internal">mlx.nn.Module.children</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.children.html#mlx.nn.Module.children"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.children()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.eval.html"
    class="reference internal">mlx.nn.Module.eval</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.eval.html#mlx.nn.Module.eval"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.eval()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.filter_and_map.html"
    class="reference internal">mlx.nn.Module.filter_and_map</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.filter_and_map.html#mlx.nn.Module.filter_and_map"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.filter_and_map()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.freeze.html"
    class="reference internal">mlx.nn.Module.freeze</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.freeze.html#mlx.nn.Module.freeze"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.freeze()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.leaf_modules.html"
    class="reference internal">mlx.nn.Module.leaf_modules</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.leaf_modules.html#mlx.nn.Module.leaf_modules"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.leaf_modules()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.load_weights.html"
    class="reference internal">mlx.nn.Module.load_weights</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.load_weights.html#mlx.nn.Module.load_weights"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.load_weights()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.modules.html"
    class="reference internal">mlx.nn.Module.modules</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.modules.html#mlx.nn.Module.modules"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.modules()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.named_modules.html"
    class="reference internal">mlx.nn.Module.named_modules</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.named_modules.html#mlx.nn.Module.named_modules"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.named_modules()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.parameters.html"
    class="reference internal">mlx.nn.Module.parameters</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.parameters()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.save_weights.html"
    class="reference internal">mlx.nn.Module.save_weights</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.save_weights.html#mlx.nn.Module.save_weights"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.save_weights()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.set_dtype.html"
    class="reference internal">mlx.nn.Module.set_dtype</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.set_dtype.html#mlx.nn.Module.set_dtype"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.set_dtype()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.train.html"
    class="reference internal">mlx.nn.Module.train</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.train.html#mlx.nn.Module.train"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.train()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.trainable_parameters.html"
    class="reference internal">mlx.nn.Module.trainable_parameters</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.trainable_parameters.html#mlx.nn.Module.trainable_parameters"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.trainable_parameters()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.unfreeze.html"
    class="reference internal">mlx.nn.Module.unfreeze</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.unfreeze.html#mlx.nn.Module.unfreeze"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.unfreeze()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update.html"
    class="reference internal">mlx.nn.Module.update</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.update()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update_modules.html"
    class="reference internal">mlx.nn.Module.update_modules</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update_modules.html#mlx.nn.Module.update_modules"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Module.update_modules()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/layers.html"
  class="reference internal">Layers</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ALiBi.html"
    class="reference internal">mlx.nn.ALiBi</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ALiBi.html#mlx.nn.ALiBi"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ALiBi</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool1d.html"
    class="reference internal">mlx.nn.AvgPool1d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool1d.html#mlx.nn.AvgPool1d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">AvgPool1d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html"
    class="reference internal">mlx.nn.AvgPool2d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html#mlx.nn.AvgPool2d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">AvgPool2d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool3d.html"
    class="reference internal">mlx.nn.AvgPool3d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool3d.html#mlx.nn.AvgPool3d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">AvgPool3d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.BatchNorm.html"
    class="reference internal">mlx.nn.BatchNorm</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.BatchNorm.html#mlx.nn.BatchNorm"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">BatchNorm</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.CELU.html"
    class="reference internal">mlx.nn.CELU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.CELU.html#mlx.nn.CELU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">CELU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv1d.html"
    class="reference internal">mlx.nn.Conv1d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv1d.html#mlx.nn.Conv1d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Conv1d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html"
    class="reference internal">mlx.nn.Conv2d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html#mlx.nn.Conv2d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Conv2d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv3d.html"
    class="reference internal">mlx.nn.Conv3d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv3d.html#mlx.nn.Conv3d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Conv3d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html"
    class="reference internal">mlx.nn.ConvTranspose1d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html#mlx.nn.ConvTranspose1d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ConvTranspose1d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose2d.html"
    class="reference internal">mlx.nn.ConvTranspose2d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose2d.html#mlx.nn.ConvTranspose2d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ConvTranspose2d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose3d.html"
    class="reference internal">mlx.nn.ConvTranspose3d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose3d.html#mlx.nn.ConvTranspose3d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ConvTranspose3d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout.html"
    class="reference internal">mlx.nn.Dropout</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout.html#mlx.nn.Dropout"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Dropout</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout2d.html"
    class="reference internal">mlx.nn.Dropout2d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout2d.html#mlx.nn.Dropout2d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Dropout2d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout3d.html"
    class="reference internal">mlx.nn.Dropout3d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout3d.html#mlx.nn.Dropout3d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Dropout3d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html"
    class="reference internal">mlx.nn.Embedding</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Embedding</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ELU.html"
    class="reference internal">mlx.nn.ELU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ELU.html#mlx.nn.ELU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ELU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GELU.html"
    class="reference internal">mlx.nn.GELU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GELU.html#mlx.nn.GELU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">GELU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GLU.html"
    class="reference internal">mlx.nn.GLU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GLU.html#mlx.nn.GLU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">GLU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GroupNorm.html"
    class="reference internal">mlx.nn.GroupNorm</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GroupNorm.html#mlx.nn.GroupNorm"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">GroupNorm</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GRU.html"
    class="reference internal">mlx.nn.GRU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GRU.html#mlx.nn.GRU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">GRU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardShrink.html"
    class="reference internal">mlx.nn.HardShrink</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardShrink.html#mlx.nn.HardShrink"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">HardShrink</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardTanh.html"
    class="reference internal">mlx.nn.HardTanh</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardTanh.html#mlx.nn.HardTanh"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">HardTanh</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Hardswish.html"
    class="reference internal">mlx.nn.Hardswish</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Hardswish.html#mlx.nn.Hardswish"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Hardswish</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.InstanceNorm.html"
    class="reference internal">mlx.nn.InstanceNorm</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.InstanceNorm.html#mlx.nn.InstanceNorm"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">InstanceNorm</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LayerNorm.html"
    class="reference internal">mlx.nn.LayerNorm</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LayerNorm.html#mlx.nn.LayerNorm"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">LayerNorm</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LeakyReLU.html"
    class="reference internal">mlx.nn.LeakyReLU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LeakyReLU.html#mlx.nn.LeakyReLU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">LeakyReLU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Linear.html"
    class="reference internal">mlx.nn.Linear</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Linear</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSigmoid.html"
    class="reference internal">mlx.nn.LogSigmoid</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSigmoid.html#mlx.nn.LogSigmoid"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">LogSigmoid</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSoftmax.html"
    class="reference internal">mlx.nn.LogSoftmax</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSoftmax.html#mlx.nn.LogSoftmax"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">LogSoftmax</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LSTM.html"
    class="reference internal">mlx.nn.LSTM</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LSTM.html#mlx.nn.LSTM"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">LSTM</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool1d.html"
    class="reference internal">mlx.nn.MaxPool1d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool1d.html#mlx.nn.MaxPool1d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">MaxPool1d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool2d.html"
    class="reference internal">mlx.nn.MaxPool2d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool2d.html#mlx.nn.MaxPool2d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">MaxPool2d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool3d.html"
    class="reference internal">mlx.nn.MaxPool3d</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool3d.html#mlx.nn.MaxPool3d"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">MaxPool3d</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Mish.html"
    class="reference internal">mlx.nn.Mish</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Mish.html#mlx.nn.Mish"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Mish</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html"
    class="reference internal">mlx.nn.MultiHeadAttention</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html#mlx.nn.MultiHeadAttention"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">MultiHeadAttention</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.PReLU.html"
    class="reference internal">mlx.nn.PReLU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.PReLU.html#mlx.nn.PReLU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">PReLU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedEmbedding.html"
    class="reference internal">mlx.nn.QuantizedEmbedding</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedEmbedding.html#mlx.nn.QuantizedEmbedding"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">QuantizedEmbedding</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html"
    class="reference internal">mlx.nn.QuantizedLinear</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html#mlx.nn.QuantizedLinear"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">QuantizedLinear</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RMSNorm.html"
    class="reference internal">mlx.nn.RMSNorm</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RMSNorm.html#mlx.nn.RMSNorm"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">RMSNorm</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ReLU.html"
    class="reference internal">mlx.nn.ReLU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ReLU.html#mlx.nn.ReLU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ReLU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ReLU6.html"
    class="reference internal">mlx.nn.ReLU6</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ReLU6.html#mlx.nn.ReLU6"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">ReLU6</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RNN.html"
    class="reference internal">mlx.nn.RNN</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RNN.html#mlx.nn.RNN"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">RNN</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html"
    class="reference internal">mlx.nn.RoPE</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html#mlx.nn.RoPE"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">RoPE</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SELU.html"
    class="reference internal">mlx.nn.SELU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SELU.html#mlx.nn.SELU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">SELU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Sequential.html"
    class="reference internal">mlx.nn.Sequential</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Sequential.html#mlx.nn.Sequential"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Sequential</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Sigmoid.html"
    class="reference internal">mlx.nn.Sigmoid</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Sigmoid.html#mlx.nn.Sigmoid"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Sigmoid</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SiLU.html"
    class="reference internal">mlx.nn.SiLU</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SiLU.html#mlx.nn.SiLU"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">SiLU</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SinusoidalPositionalEncoding.html"
    class="reference internal">mlx.nn.SinusoidalPositionalEncoding</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SinusoidalPositionalEncoding.html#mlx.nn.SinusoidalPositionalEncoding"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">SinusoidalPositionalEncoding</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softmin.html"
    class="reference internal">mlx.nn.Softmin</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softmin.html#mlx.nn.Softmin"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Softmin</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softshrink.html"
    class="reference internal">mlx.nn.Softshrink</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softshrink.html#mlx.nn.Softshrink"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Softshrink</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softsign.html"
    class="reference internal">mlx.nn.Softsign</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softsign.html#mlx.nn.Softsign"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Softsign</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softmax.html"
    class="reference internal">mlx.nn.Softmax</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softmax.html#mlx.nn.Softmax"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Softmax</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softplus.html"
    class="reference internal">mlx.nn.Softplus</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softplus.html#mlx.nn.Softplus"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Softplus</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Step.html"
    class="reference internal">mlx.nn.Step</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Step.html#mlx.nn.Step"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Step</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Tanh.html"
    class="reference internal">mlx.nn.Tanh</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Tanh.html#mlx.nn.Tanh"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Tanh</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Transformer.html"
    class="reference internal">mlx.nn.Transformer</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Transformer.html#mlx.nn.Transformer"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Transformer</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Upsample.html"
    class="reference internal">mlx.nn.Upsample</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Upsample.html#mlx.nn.Upsample"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">Upsample</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/functions.html"
  class="reference internal">Functions</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.elu.html"
    class="reference internal">mlx.nn.elu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.elu.html#mlx.nn.elu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">elu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.celu.html"
    class="reference internal">mlx.nn.celu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.celu.html#mlx.nn.celu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">celu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu.html"
    class="reference internal">mlx.nn.gelu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu.html#mlx.nn.gelu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">gelu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html"
    class="reference internal">mlx.nn.gelu_approx</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html#mlx.nn.gelu_approx"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">gelu_approx</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_fast_approx.html"
    class="reference internal">mlx.nn.gelu_fast_approx</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_fast_approx.html#mlx.nn.gelu_fast_approx"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">gelu_fast_approx</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.glu.html"
    class="reference internal">mlx.nn.glu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.glu.html#mlx.nn.glu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">glu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hard_shrink.html"
    class="reference internal">mlx.nn.hard_shrink</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hard_shrink.html#mlx.nn.hard_shrink"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">hard_shrink</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hard_tanh.html"
    class="reference internal">mlx.nn.hard_tanh</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hard_tanh.html#mlx.nn.hard_tanh"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">hard_tanh</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hardswish.html"
    class="reference internal">mlx.nn.hardswish</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.hardswish.html#mlx.nn.hardswish"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">hardswish</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.leaky_relu.html"
    class="reference internal">mlx.nn.leaky_relu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.leaky_relu.html#mlx.nn.leaky_relu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">leaky_relu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.log_sigmoid.html"
    class="reference internal">mlx.nn.log_sigmoid</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.log_sigmoid.html#mlx.nn.log_sigmoid"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">log_sigmoid</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.log_softmax.html"
    class="reference internal">mlx.nn.log_softmax</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.log_softmax.html#mlx.nn.log_softmax"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">log_softmax</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.mish.html"
    class="reference internal">mlx.nn.mish</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.mish.html#mlx.nn.mish"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">mish</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.prelu.html"
    class="reference internal">mlx.nn.prelu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.prelu.html#mlx.nn.prelu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">prelu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu.html"
    class="reference internal">mlx.nn.relu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu.html#mlx.nn.relu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">relu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu6.html"
    class="reference internal">mlx.nn.relu6</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu6.html#mlx.nn.relu6"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">relu6</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.selu.html"
    class="reference internal">mlx.nn.selu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.selu.html#mlx.nn.selu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">selu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.sigmoid.html"
    class="reference internal">mlx.nn.sigmoid</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.sigmoid.html#mlx.nn.sigmoid"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">sigmoid</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.silu.html"
    class="reference internal">mlx.nn.silu</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.silu.html#mlx.nn.silu"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">silu</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softmax.html"
    class="reference internal">mlx.nn.softmax</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softmax.html#mlx.nn.softmax"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">softmax</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softmin.html"
    class="reference internal">mlx.nn.softmin</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softmin.html#mlx.nn.softmin"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">softmin</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softplus.html"
    class="reference internal">mlx.nn.softplus</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softplus.html#mlx.nn.softplus"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">softplus</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softshrink.html"
    class="reference internal">mlx.nn.softshrink</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.softshrink.html#mlx.nn.softshrink"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">softshrink</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.step.html"
    class="reference internal">mlx.nn.step</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.step.html#mlx.nn.step"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">step</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.tanh.html"
    class="reference internal">mlx.nn.tanh</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.tanh.html#mlx.nn.tanh"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">tanh</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/losses.html"
  class="reference internal">Loss Functions</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.binary_cross_entropy.html"
    class="reference internal">mlx.nn.losses.binary_cross_entropy</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.binary_cross_entropy.html#mlx.nn.losses.binary_cross_entropy"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">binary_cross_entropy</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cosine_similarity_loss.html"
    class="reference internal">mlx.nn.losses.cosine_similarity_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cosine_similarity_loss.html#mlx.nn.losses.cosine_similarity_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">cosine_similarity_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html"
    class="reference internal">mlx.nn.losses.cross_entropy</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html#mlx.nn.losses.cross_entropy"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">cross_entropy</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html"
    class="reference internal">mlx.nn.losses.gaussian_nll_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html#mlx.nn.losses.gaussian_nll_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">gaussian_nll_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.hinge_loss.html"
    class="reference internal">mlx.nn.losses.hinge_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.hinge_loss.html#mlx.nn.losses.hinge_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">hinge_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.huber_loss.html"
    class="reference internal">mlx.nn.losses.huber_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.huber_loss.html#mlx.nn.losses.huber_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">huber_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.kl_div_loss.html"
    class="reference internal">mlx.nn.losses.kl_div_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.kl_div_loss.html#mlx.nn.losses.kl_div_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">kl_div_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.l1_loss.html"
    class="reference internal">mlx.nn.losses.l1_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.l1_loss.html#mlx.nn.losses.l1_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">l1_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.log_cosh_loss.html"
    class="reference internal">mlx.nn.losses.log_cosh_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.log_cosh_loss.html#mlx.nn.losses.log_cosh_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">log_cosh_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.margin_ranking_loss.html"
    class="reference internal">mlx.nn.losses.margin_ranking_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.margin_ranking_loss.html#mlx.nn.losses.margin_ranking_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">margin_ranking_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.mse_loss.html"
    class="reference internal">mlx.nn.losses.mse_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.mse_loss.html#mlx.nn.losses.mse_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">mse_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.nll_loss.html"
    class="reference internal">mlx.nn.losses.nll_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.nll_loss.html#mlx.nn.losses.nll_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">nll_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.smooth_l1_loss.html"
    class="reference internal">mlx.nn.losses.smooth_l1_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.smooth_l1_loss.html#mlx.nn.losses.smooth_l1_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">smooth_l1_loss</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html"
    class="reference internal">mlx.nn.losses.triplet_loss</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html#mlx.nn.losses.triplet_loss"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">triplet_loss</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/init.html"
  class="reference internal">Initializers</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.constant.html"
    class="reference internal">mlx.nn.init.constant</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.constant.html#mlx.nn.init.constant"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">constant()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.normal.html"
    class="reference internal">mlx.nn.init.normal</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.normal.html#mlx.nn.init.normal"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">normal()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.uniform.html"
    class="reference internal">mlx.nn.init.uniform</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.uniform.html#mlx.nn.init.uniform"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">uniform()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.identity.html"
    class="reference internal">mlx.nn.init.identity</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.identity.html#mlx.nn.init.identity"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">identity()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_normal.html"
    class="reference internal">mlx.nn.init.glorot_normal</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_normal.html#mlx.nn.init.glorot_normal"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">glorot_normal()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_uniform.html"
    class="reference internal">mlx.nn.init.glorot_uniform</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_uniform.html#mlx.nn.init.glorot_uniform"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">glorot_uniform()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_normal.html"
    class="reference internal">mlx.nn.init.he_normal</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_normal.html#mlx.nn.init.he_normal"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">he_normal()</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_uniform.html"
    class="reference internal">mlx.nn.init.he_uniform</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.he_uniform.html#mlx.nn.init.he_uniform"
      class="reference internal"><span class="pre"><code
      class="docutils literal notranslate">he_uniform()</code></span></a>

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clear_cache.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.clear_cache

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.value_and_grad

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
  href="https://ml-explore.github.io/mlx/build/html/#quick-start-with-neural-networks"
  class="reference internal nav-link">Quick Start with Neural Networks</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#the-module-class"
  class="reference internal nav-link">The Module Class</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#parameters"
    class="reference internal nav-link">Parameters</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#updating-the-parameters"
    class="reference internal nav-link">Updating the Parameters</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#inspecting-modules"
    class="reference internal nav-link">Inspecting Modules</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#value-and-grad"
  class="reference internal nav-link">Value and Grad</a>

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
