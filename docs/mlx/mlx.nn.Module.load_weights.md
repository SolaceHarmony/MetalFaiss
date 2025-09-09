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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Module.load_weights.rst"
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

# mlx.nn.Module.load_weights

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.load_weights"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module.load_weights()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-module-load-weights" class="section">

# mlx.nn.Module.load_weights<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-module-load-weights"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-prename descclassname"><span class="pre">Module.</span></span><span class="sig-name descname"><span class="pre">load_weights</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">file_or_weights</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.List"
class="reference external" title="(in Python v3.13)"><span
class="pre">List</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/typing.html#typing.Tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">Tuple</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">strict</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.layers.base.Module"><span
class="pre">Module</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.load_weights"
class="headerlink" title="Link to this definition">#</a>  
Update the model’s weights from a <span class="pre">`.npz`</span>, a
<span class="pre">`.safetensors`</span> file, or a list.

Parameters<span class="colon">:</span>  
- **file_or_weights**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  *mx.array))*) – The path to the weights
  <span class="pre">`.npz`</span> file (<span class="pre">`.npz`</span>
  or <span class="pre">`.safetensors`</span>) or a list of pairs of
  parameter names and arrays.

- **strict**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If <span class="pre">`True`</span> then checks that the
  provided weights exactly match the parameters of the model. Otherwise,
  only the weights actually contained in the model are loaded and shapes
  are not checked. Default: <span class="pre">`True`</span>.

Returns<span class="colon">:</span>  
The module instance after updating the weights.

Example

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn
    model = nn.Linear(10, 10)

    # Load from file
    model.load_weights("weights.npz")

    # Load from .safetensors file
    model.load_weights("weights.safetensors")

    # Load from list
    weights = [
        ("weight", mx.random.uniform(shape=(10, 10))),
        ("bias",  mx.zeros((10,))),
    ]
    model.load_weights(weights)

    # Missing weight
    weights = [
        ("weight", mx.random.uniform(shape=(10, 10))),
    ]

    # Raises a ValueError exception
    model.load_weights(weights)

    # Ok, only updates the weight but not the bias
    model.load_weights(weights, strict=False)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.leaf_modules.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Module.leaf_modules

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.modules.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.Module.modules

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.load_weights"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module.load_weights()</code></span></a>

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
