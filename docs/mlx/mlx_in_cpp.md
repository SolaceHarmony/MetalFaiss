Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (mlx_in_cpp.md):
- C++ usage with CMake; includes example main and CMakeLists.
- Add curated build tips: include dirs, linkage, and Metal availability.
-->

## Curated Notes

- Ensure your `find_package(MLX CONFIG REQUIRED)` (or equivalent) points to the installed MLX C++ package; set `CMAKE_PREFIX_PATH` if needed.
- Link against the MLX targets provided by the package (e.g., `target_link_libraries(example PRIVATE mlx::mlx)` if exported).
- On non‑Metal builds (CPU‑only), guard GPU‑dependent code with availability checks to keep the binary portable across hosts.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/dev/mlx_in_cpp.rst"
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

# Using MLX in C++

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="using-mlx-in-c" class="section">

<span id="mlx-in-cpp"></span>

# Using MLX in C++<a href="https://ml-explore.github.io/mlx/build/html/#using-mlx-in-c"
class="headerlink" title="Link to this heading">#</a>

You can use MLX in a C++ project with CMake.

<div class="admonition note">

Note

This guide is based one the following <a
href="https://github.com/ml-explore/mlx/tree/main/examples/cmake_project"
class="reference external">example using MLX in C++</a>

</div>

First install MLX:

<div class="highlight-bash notranslate">

<div class="highlight">

    pip install -U mlx

</div>

</div>

You can also install the MLX Python package from source or just the C++
library. For more information see the <a
href="https://ml-explore.github.io/mlx/build/html/install.html#build-and-install"
class="reference internal"><span class="std std-ref">documentation on
installing MLX</span></a>.

Next make an example program in <span class="pre">`example.cpp`</span>:

<div class="highlight-C++ notranslate">

<div class="highlight">

    #include <iostream>

    #include "mlx/mlx.h"

    namespace mx = mlx::core;

    int main() {
      auto x = mx::array({1, 2, 3});
      auto y = mx::array({1, 2, 3});
      std::cout << x + y << std::endl;
      return 0;
    }

</div>

</div>

The next step is to setup a CMake file in
<span class="pre">`CMakeLists.txt`</span>:

<div class="highlight-cmake notranslate">

<div class="highlight">

    cmake_minimum_required(VERSION 3.27)

    project(example LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

</div>

</div>

Depending on how you installed MLX, you may need to tell CMake where to
find it.

If you installed MLX with Python, then add the following to the CMake
file:

<div class="highlight-cmake notranslate">

<div class="highlight">

    find_package(
      Python 3.9
      COMPONENTS Interpreter Development.Module
      REQUIRED)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE MLX_ROOT)

</div>

</div>

If you installed the MLX C++ package to a system path, then CMake should
be able to find it. If you installed it to a non-standard location or
CMake can’t find MLX then set <span class="pre">`MLX_ROOT`</span> to the
location where MLX is installed:

<div class="highlight-cmake notranslate">

<div class="highlight">

    set(MLX_ROOT "/path/to/mlx/")

</div>

</div>

Next, instruct CMake to find MLX:

<div class="highlight-cmake notranslate">

<div class="highlight">

    find_package(MLX CONFIG REQUIRED)

</div>

</div>

Finally, add the <span class="pre">`example.cpp`</span> program as an
executable and link MLX.

<div class="highlight-cmake notranslate">

<div class="highlight">

    add_executable(example example.cpp)
    target_link_libraries(example PRIVATE mlx)

</div>

</div>

You can build the example with:

<div class="highlight-bash notranslate">

<div class="highlight">

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build

</div>

</div>

And run it with:

<div class="highlight-bash notranslate">

<div class="highlight">

    ./build/example

</div>

</div>

Note
<span class="pre">`find_package(MLX`</span>` `<span class="pre">`CONFIG`</span>` `<span class="pre">`REQUIRED)`</span>
sets the following variables:

<div class="pst-scrollable-table-container">

| Variable | Description |
|----|----|
| MLX_FOUND | <span class="pre">`True`</span> if MLX is found |
| MLX_INCLUDE_DIRS | Include directory |
| MLX_LIBRARIES | Libraries to link against |
| MLX_CXX_FLAGS | Additional compiler flags |
| MLX_BUILD_ACCELERATE | <span class="pre">`True`</span> if MLX was built with Accelerate |
| MLX_BUILD_METAL | <span class="pre">`True`</span> if MLX was built with Metal |

<span class="caption-text">Package
Variables</span><a href="https://ml-explore.github.io/mlx/build/html/#id1"
class="headerlink" title="Link to this table">#</a> {#id1}

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Custom Metal Kernels

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
