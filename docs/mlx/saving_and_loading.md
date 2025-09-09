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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/saving_and_loading.rst"
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

# Saving and Loading Arrays

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="saving-and-loading-arrays" class="section">

<span id="saving-and-loading"></span>

# Saving and Loading Arrays<a
href="https://ml-explore.github.io/mlx/build/html/#saving-and-loading-arrays"
class="headerlink" title="Link to this heading">#</a>

MLX supports multiple array serialization formats.

<div class="pst-scrollable-table-container">

| Format | Extension | Function | Notes |
|----|----|----|----|
| NumPy | <span class="pre">`.npy`</span> | <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save.html#mlx.core.save"
class="reference internal" title="mlx.core.save"><span class="pre"><code
class="sourceCode python">save()</code></span></a> | Single arrays only |
| NumPy archive | <span class="pre">`.npz`</span> | <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.savez.html#mlx.core.savez"
class="reference internal" title="mlx.core.savez"><span
class="pre"><code class="sourceCode python">savez()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.savez_compressed.html#mlx.core.savez_compressed"
class="reference internal" title="mlx.core.savez_compressed"><span
class="pre"><code
class="sourceCode python">savez_compressed()</code></span></a> | Multiple arrays |
| Safetensors | <span class="pre">`.safetensors`</span> | <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save_safetensors.html#mlx.core.save_safetensors"
class="reference internal" title="mlx.core.save_safetensors"><span
class="pre"><code
class="sourceCode python">save_safetensors()</code></span></a> | Multiple arrays |
| GGUF | <span class="pre">`.gguf`</span> | <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save_gguf.html#mlx.core.save_gguf"
class="reference internal" title="mlx.core.save_gguf"><span
class="pre"><code
class="sourceCode python">save_gguf()</code></span></a> | Multiple arrays |

<span class="caption-text">Serialization
Formats</span><a href="https://ml-explore.github.io/mlx/build/html/#id1"
class="headerlink" title="Link to this table">#</a> {#id1}

</div>

The <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html#mlx.core.load"
class="reference internal" title="mlx.core.load"><span class="pre"><code
class="sourceCode python">load()</code></span></a> function will load
any of the supported serialization formats. It determines the format
from the extensions. The output of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html#mlx.core.load"
class="reference internal" title="mlx.core.load"><span class="pre"><code
class="sourceCode python">load()</code></span></a> depends on the
format.

Here’s an example of saving a single array to a file:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> a = mx.array([1.0])
    >>> mx.save("array", a)

</div>

</div>

The array <span class="pre">`a`</span> will be saved in the file
<span class="pre">`array.npy`</span> (notice the extension is
automatically added). Including the extension is optional; if it is
missing it will be added. You can load the array with:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> mx.load("array.npy")
    array([1], dtype=float32)

</div>

</div>

Here’s an example of saving several arrays to a single file:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> a = mx.array([1.0])
    >>> b = mx.array([2.0])
    >>> mx.savez("arrays", a, b=b)

</div>

</div>

For compatibility with <a
href="https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez"
class="reference external" title="(in NumPy v2.2)"><span
class="pre"><code
class="sourceCode python">numpy.savez()</code></span></a> the MLX <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.savez.html#mlx.core.savez"
class="reference internal" title="mlx.core.savez"><span
class="pre"><code class="sourceCode python">savez()</code></span></a>
takes arrays as arguments. If the keywords are missing, then default
names will be provided. This can be loaded with:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> mx.load("arrays.npz")
    {'b': array([2], dtype=float32), 'arr_0': array([1], dtype=float32)}

</div>

</div>

In this case <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html#mlx.core.load"
class="reference internal" title="mlx.core.load"><span class="pre"><code
class="sourceCode python">load()</code></span></a> returns a dictionary
of names to arrays.

The functions <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save_safetensors.html#mlx.core.save_safetensors"
class="reference internal" title="mlx.core.save_safetensors"><span
class="pre"><code
class="sourceCode python">save_safetensors()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save_gguf.html#mlx.core.save_gguf"
class="reference internal" title="mlx.core.save_gguf"><span
class="pre"><code
class="sourceCode python">save_gguf()</code></span></a> are similar to
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.savez.html#mlx.core.savez"
class="reference internal" title="mlx.core.savez"><span
class="pre"><code class="sourceCode python">savez()</code></span></a>,
but they take as input a
<a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">dict</span></code></span></a>
of string names to arrays:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> a = mx.array([1.0])
    >>> b = mx.array([2.0])
    >>> mx.save_safetensors("arrays", {"a": a, "b": b})

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/indexing.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Indexing Arrays

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Function Transforms

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
