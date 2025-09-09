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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/module.rst"
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

# Module

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="module" class="section">

# Module<a href="https://ml-explore.github.io/mlx/build/html/#module"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Module</span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module"
class="headerlink" title="Link to this definition">#</a>  
Base class for building neural networks with MLX.

All the layers provided in <span class="pre">`mlx.nn.layers`</span>
subclass this class and your models should do the same.

A <span class="pre">`Module`</span> can contain other
<span class="pre">`Module`</span> instances or <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> instances in
arbitrary nesting of python lists or dicts. The
<span class="pre">`Module`</span> then allows recursively extracting all
the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> instances
using <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters"
class="reference internal" title="mlx.nn.Module.parameters"><span
class="pre"><code
class="sourceCode python">mlx.nn.Module.parameters()</code></span></a>.

In addition, the <span class="pre">`Module`</span> has the concept of
trainable and non trainable parameters (called “frozen”). When using <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
class="reference internal" title="mlx.nn.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.nn.value_and_grad()</code></span></a> the
gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the
“frozen” set by calling <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.freeze.html#mlx.nn.Module.freeze"
class="reference internal" title="mlx.nn.Module.freeze"><span
class="pre"><code class="sourceCode python">freeze()</code></span></a>.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())

</div>

</div>

Attributes

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.training.html#mlx.nn.Module.training"
class="reference internal" title="mlx.nn.Module.training"><span
class="pre"><code
class="sourceCode python">Module.training</code></span></a> | Boolean indicating if the model is in training mode. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.state.html#mlx.nn.Module.state"
class="reference internal" title="mlx.nn.Module.state"><span
class="pre"><code
class="sourceCode python">Module.state</code></span></a> | The module's state dictionary |

</div>

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.apply.html#mlx.nn.Module.apply"
class="reference internal" title="mlx.nn.Module.apply"><span
class="pre"><code
class="sourceCode python">Module.<span class="bu">apply</span></code></span></a>(map_fn\[, filter_fn\]) | Map all the parameters using the provided <span class="pre">`map_fn`</span> and immediately update the module with the mapped parameters. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.apply_to_modules.html#mlx.nn.Module.apply_to_modules"
class="reference internal" title="mlx.nn.Module.apply_to_modules"><span
class="pre"><code
class="sourceCode python">Module.apply_to_modules</code></span></a>(apply_fn) | Apply a function to all the modules in this instance (including this instance). |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.children.html#mlx.nn.Module.children"
class="reference internal" title="mlx.nn.Module.children"><span
class="pre"><code
class="sourceCode python">Module.children</code></span></a>() | Return the direct descendants of this Module instance. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.eval.html#mlx.nn.Module.eval"
class="reference internal" title="mlx.nn.Module.eval"><span
class="pre"><code
class="sourceCode python">Module.<span class="bu">eval</span></code></span></a>() | Set the model to evaluation mode. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.filter_and_map.html#mlx.nn.Module.filter_and_map"
class="reference internal" title="mlx.nn.Module.filter_and_map"><span
class="pre"><code
class="sourceCode python">Module.filter_and_map</code></span></a>(filter_fn\[, map_fn, ...\]) | Recursively filter the contents of the module using <span class="pre">`filter_fn`</span>, namely only select keys and values where <span class="pre">`filter_fn`</span> returns true. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.freeze.html#mlx.nn.Module.freeze"
class="reference internal" title="mlx.nn.Module.freeze"><span
class="pre"><code
class="sourceCode python">Module.freeze</code></span></a>(\*\[, recurse, keys, strict\]) | Freeze the Module's parameters or some of them. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.leaf_modules.html#mlx.nn.Module.leaf_modules"
class="reference internal" title="mlx.nn.Module.leaf_modules"><span
class="pre"><code
class="sourceCode python">Module.leaf_modules</code></span></a>() | Return the submodules that do not contain other modules. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.load_weights.html#mlx.nn.Module.load_weights"
class="reference internal" title="mlx.nn.Module.load_weights"><span
class="pre"><code
class="sourceCode python">Module.load_weights</code></span></a>(file_or_weights\[, strict\]) | Update the model's weights from a <span class="pre">`.npz`</span>, a <span class="pre">`.safetensors`</span> file, or a list. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.modules.html#mlx.nn.Module.modules"
class="reference internal" title="mlx.nn.Module.modules"><span
class="pre"><code
class="sourceCode python">Module.modules</code></span></a>() | Return a list with all the modules in this instance. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.named_modules.html#mlx.nn.Module.named_modules"
class="reference internal" title="mlx.nn.Module.named_modules"><span
class="pre"><code
class="sourceCode python">Module.named_modules</code></span></a>() | Return a list with all the modules in this instance and their name with dot notation. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters"
class="reference internal" title="mlx.nn.Module.parameters"><span
class="pre"><code
class="sourceCode python">Module.parameters</code></span></a>() | Recursively return all the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> members of this Module as a dict of dicts and lists. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.save_weights.html#mlx.nn.Module.save_weights"
class="reference internal" title="mlx.nn.Module.save_weights"><span
class="pre"><code
class="sourceCode python">Module.save_weights</code></span></a>(file) | Save the model's weights to a file. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.set_dtype.html#mlx.nn.Module.set_dtype"
class="reference internal" title="mlx.nn.Module.set_dtype"><span
class="pre"><code
class="sourceCode python">Module.set_dtype</code></span></a>(dtype\[, predicate\]) | Set the dtype of the module's parameters. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.train.html#mlx.nn.Module.train"
class="reference internal" title="mlx.nn.Module.train"><span
class="pre"><code
class="sourceCode python">Module.train</code></span></a>(\[mode\]) | Set the model in or out of training mode. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.trainable_parameters.html#mlx.nn.Module.trainable_parameters"
class="reference internal"
title="mlx.nn.Module.trainable_parameters"><span class="pre"><code
class="sourceCode python">Module.trainable_parameters</code></span></a>() | Recursively return all the non frozen <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> members of this Module as a dict of dicts and lists. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.unfreeze.html#mlx.nn.Module.unfreeze"
class="reference internal" title="mlx.nn.Module.unfreeze"><span
class="pre"><code
class="sourceCode python">Module.unfreeze</code></span></a>(\*\[, recurse, keys, strict\]) | Unfreeze the Module's parameters or some of them. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update.html#mlx.nn.Module.update"
class="reference internal" title="mlx.nn.Module.update"><span
class="pre"><code
class="sourceCode python">Module.update</code></span></a>(parameters) | Replace the parameters of this Module with the provided ones in the dict of dicts and lists. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update_modules.html#mlx.nn.Module.update_modules"
class="reference internal" title="mlx.nn.Module.update_modules"><span
class="pre"><code
class="sourceCode python">Module.update_modules</code></span></a>(modules) | Replace the child modules of this <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">Module</code></span></a> instance with the provided ones in the dict of dicts and lists. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.average_gradients.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.average_gradients

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.training.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.Module.training

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module</code></span></a>

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
