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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.DtypeCategory.rst"
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

# mlx.core.DtypeCategory

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.DtypeCategory"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">DtypeCategory</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#mlx.core.DtypeCategory.__init__"
    class="reference internal nav-link"><span class="pre"><code
    class="docutils literal notranslate">DtypeCategory.__init__()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-dtypecategory" class="section">

# mlx.core.DtypeCategory<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-dtypecategory"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">DtypeCategory</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">value</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.DtypeCategory"
class="headerlink" title="Link to this definition">#</a>  
Type to hold categories of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">dtypes</code></span></a>.

- <span class="pre">`generic`</span>

  - <a
    href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
    class="reference internal"><span class="std std-ref">bool_</span></a>

  - <span class="pre">`number`</span>

    - <span class="pre">`integer`</span>

      - <span class="pre">`unsignedinteger`</span>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">uint8</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">uint16</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">uint32</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">uint64</span></a>

      - <span class="pre">`signedinteger`</span>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">int8</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">int32</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">int64</span></a>

    - <span class="pre">`inexact`</span>

      - <span class="pre">`floating`</span>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">float16</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">bfloat16</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">float32</span></a>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span class="std std-ref">float64</span></a>

      - <span class="pre">`complexfloating`</span>

        - <a
          href="https://ml-explore.github.io/mlx/build/html/python/data_types.html#data-types"
          class="reference internal"><span
          class="std std-ref">complex64</span></a>

See also <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.issubdtype.html#mlx.core.issubdtype"
class="reference internal" title="mlx.core.issubdtype"><span
class="pre"><code
class="sourceCode python">issubdtype()</code></span></a>.

<span class="sig-name descname"><span class="pre">\_\_init\_\_</span></span><span class="sig-paren">(</span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.DtypeCategory.__init__"
class="headerlink" title="Link to this definition">#</a>  

Attributes

<div class="pst-scrollable-table-container">

|                                            |     |
|--------------------------------------------|-----|
| <span class="pre">`complexfloating`</span> |     |
| <span class="pre">`floating`</span>        |     |
| <span class="pre">`inexact`</span>         |     |
| <span class="pre">`signedinteger`</span>   |     |
| <span class="pre">`unsignedinteger`</span> |     |
| <span class="pre">`integer`</span>         |     |
| <span class="pre">`number`</span>          |     |
| <span class="pre">`generic`</span>         |     |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.Dtype

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.issubdtype.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.issubdtype

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.DtypeCategory"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">DtypeCategory</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#mlx.core.DtypeCategory.__init__"
    class="reference internal nav-link"><span class="pre"><code
    class="docutils literal notranslate">DtypeCategory.__init__()</code></span></a>

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
