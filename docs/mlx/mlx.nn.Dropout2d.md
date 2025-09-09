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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Dropout2d.rst"
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

# mlx.nn.Dropout2d

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Dropout2d"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Dropout2d</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-dropout2d" class="section">

# mlx.nn.Dropout2d<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-dropout2d"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Dropout2d</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">p</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0.5</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Dropout2d"
class="headerlink" title="Link to this definition">#</a>  
Apply 2D channel-wise dropout during training.

Randomly zero out entire channels independently with probability
<span class="math notranslate nohighlight">\\p\\</span>. This layer
expects the channels to be last, i.e. the input shape should be
<span class="pre">`NWHC`</span> or <span class="pre">`WHC`</span>
where:<span class="pre">`N`</span> is the batch dimension,\`\`H\`\` is
the input image height,\`\`W\`\` is the input image width, and\`\`C\`\`
is the number of input channels

The remaining channels are scaled by
<span class="math notranslate nohighlight">\\\frac{1}{1-p}\\</span> to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This
is beneficial for early convolution layers where adjacent pixels are
correlated. In such case, traditional dropout may not effectively
regularize activations. For more details, see \[1\].

\[1\]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler
C., 2015. Efficient Object Localization Using Convolutional Networks.
CVPR 2015.

Parameters<span class="colon">:</span>  
**p** (<a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><em>float</em></a>)
– Probability of zeroing a channel during training.

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Dropout

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout3d.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.Dropout3d

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Dropout2d"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Dropout2d</code></span></a>

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
