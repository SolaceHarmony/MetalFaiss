Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (install.md):
- Installation prerequisites and troubleshooting for Apple Silicon.
- Helpful additions: confirm native Python, Homebrew tips, virtualenv hygiene.
-->

## Curated Notes

- Verify native Python: `python -c "import platform; print(platform.processor())"` should print `arm` on Apple Silicon.
- Prefer a clean virtual environment (venv/conda) per project; avoid mixing system and user site‑packages.
- If Homebrew installed Python, ensure your shell picks the ARM binary (e.g., avoid running under Rosetta/x86 shells).


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/install.rst"
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

# Build and Install

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#python-installation"
  class="reference internal nav-link">Python Installation</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#troubleshooting"
    class="reference internal nav-link">Troubleshooting</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#build-from-source"
  class="reference internal nav-link">Build from source</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#build-requirements"
    class="reference internal nav-link">Build Requirements</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#python-api"
    class="reference internal nav-link">Python API</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#c-api"
    class="reference internal nav-link">C++ API</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/#binary-size-minimization"
      class="reference internal nav-link">Binary Size Minimization</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#id3"
    class="reference internal nav-link">Troubleshooting</a>
    - <a href="https://ml-explore.github.io/mlx/build/html/#metal-not-found"
      class="reference internal nav-link">Metal not found</a>
    - <a href="https://ml-explore.github.io/mlx/build/html/#x86-shell"
      class="reference internal nav-link">x86 Shell</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="build-and-install" class="section">

<span id="id1"></span>

# Build and Install<a href="https://ml-explore.github.io/mlx/build/html/#build-and-install"
class="headerlink" title="Link to this heading">#</a>

<div id="python-installation" class="section">

## Python Installation<a
href="https://ml-explore.github.io/mlx/build/html/#python-installation"
class="headerlink" title="Link to this heading">#</a>

MLX is available on PyPI. All you have to do to use MLX with your own
Apple silicon computer is

<div class="highlight-shell notranslate">

<div class="highlight">

    pip install mlx

</div>

</div>

To install from PyPI you must meet the following requirements:

- Using an M series chip (Apple silicon)

- Using a native Python \>= 3.9

- macOS \>= 13.5

<div class="admonition note">

Note

MLX is only available on devices running macOS \>= 13.5 It is highly
recommended to use macOS 14 (Sonoma)

</div>

MLX is also available on conda-forge. To install MLX with conda do:

<div class="highlight-shell notranslate">

<div class="highlight">

    conda install conda-forge::mlx

</div>

</div>

<div id="troubleshooting" class="section">

### Troubleshooting<a href="https://ml-explore.github.io/mlx/build/html/#troubleshooting"
class="headerlink" title="Link to this heading">#</a>

*My OS and Python versions are in the required range but pip still does
not find a matching distribution.*

Probably you are using a non-native Python. The output of

<div class="highlight-shell notranslate">

<div class="highlight">

    python -c "import platform; print(platform.processor())"

</div>

</div>

should be <span class="pre">`arm`</span>. If it is
<span class="pre">`i386`</span> (and you have M series machine) then you
are using a non-native Python. Switch your Python to a native Python. A
good way to do this is with
<a href="https://stackoverflow.com/q/65415996"
class="reference external">Conda</a>.

</div>

</div>

<div id="build-from-source" class="section">

## Build from source<a href="https://ml-explore.github.io/mlx/build/html/#build-from-source"
class="headerlink" title="Link to this heading">#</a>

<div id="build-requirements" class="section">

### Build Requirements<a
href="https://ml-explore.github.io/mlx/build/html/#build-requirements"
class="headerlink" title="Link to this heading">#</a>

- A C++ compiler with C++17 support (e.g. Clang \>= 5.0)

- <a href="https://cmake.org/" class="reference external">cmake</a> –
  version 3.25 or later, and <span class="pre">`make`</span>

- Xcode \>= 15.0 and macOS SDK \>= 14.0

<div class="admonition note">

Note

Ensure your shell environment is native <span class="pre">`arm`</span>,
not <span class="pre">`x86`</span> via Rosetta. If the output of
<span class="pre">`uname`</span>` `<span class="pre">`-p`</span> is
<span class="pre">`x86`</span>, see the
<a href="https://ml-explore.github.io/mlx/build/html/#build-shell"
class="reference internal"><span class="std std-ref">troubleshooting
section</span></a> below.

</div>

</div>

<div id="python-api" class="section">

### Python API<a href="https://ml-explore.github.io/mlx/build/html/#python-api"
class="headerlink" title="Link to this heading">#</a>

To build and install the MLX python library from source, first, clone
MLX from <a href="https://github.com/ml-explore/mlx"
class="reference external">its GitHub repo</a>:

<div class="highlight-shell notranslate">

<div class="highlight">

    git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

</div>

</div>

Then simply build and install MLX using pip:

<div class="highlight-shell notranslate">

<div class="highlight">

    CMAKE_BUILD_PARALLEL_LEVEL=8 pip install .

</div>

</div>

For developing, install the package with development dependencies, and
use an editable install:

<div class="highlight-shell notranslate">

<div class="highlight">

    CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e ".[dev]"

</div>

</div>

Once the development dependencies are installed, you can build faster
with:

<div class="highlight-shell notranslate">

<div class="highlight">

    CMAKE_BUILD_PARALLEL_LEVEL=8 python setup.py build_ext --inplace

</div>

</div>

Run the tests with:

<div class="highlight-shell notranslate">

<div class="highlight">

    python -m unittest discover python/tests

</div>

</div>

Optional: Install stubs to enable auto completions and type checking
from your IDE:

<div class="highlight-shell notranslate">

<div class="highlight">

    python setup.py generate_stubs

</div>

</div>

</div>

<div id="c-api" class="section">

### C++ API<a href="https://ml-explore.github.io/mlx/build/html/#c-api"
class="headerlink" title="Link to this heading">#</a>

Currently, MLX must be built and installed from source.

Similarly to the python library, to build and install the MLX C++
library start by cloning MLX from
<a href="https://github.com/ml-explore/mlx"
class="reference external">its GitHub repo</a>:

<div class="highlight-shell notranslate">

<div class="highlight">

    git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

</div>

</div>

Create a build directory and run CMake and make:

<div class="highlight-shell notranslate">

<div class="highlight">

    mkdir -p build && cd build
    cmake .. && make -j

</div>

</div>

Run tests with:

<div class="highlight-shell notranslate">

<div class="highlight">

    make test

</div>

</div>

Install with:

<div class="highlight-shell notranslate">

<div class="highlight">

    make install

</div>

</div>

Note that the built <span class="pre">`mlx.metallib`</span> file should
be either at the same directory as the executable statically linked to
<span class="pre">`libmlx.a`</span> or the preprocessor constant
<span class="pre">`METAL_PATH`</span> should be defined at build time
and it should point to the path to the built metal library.

<div class="pst-scrollable-table-container">

| Option                    | Default |
|---------------------------|---------|
| MLX_BUILD_TESTS           | ON      |
| MLX_BUILD_EXAMPLES        | OFF     |
| MLX_BUILD_BENCHMARKS      | OFF     |
| MLX_BUILD_METAL           | ON      |
| MLX_BUILD_CPU             | ON      |
| MLX_BUILD_PYTHON_BINDINGS | OFF     |
| MLX_METAL_DEBUG           | OFF     |
| MLX_BUILD_SAFETENSORS     | ON      |
| MLX_BUILD_GGUF            | ON      |
| MLX_METAL_JIT             | OFF     |

<span class="caption-text">Build
Options</span><a href="https://ml-explore.github.io/mlx/build/html/#id4"
class="headerlink" title="Link to this table">#</a> {#id4}

</div>

<div class="admonition note">

Note

If you have multiple Xcode installations and wish to use a specific one
while building, you can do so by adding the following environment
variable before building

<div class="highlight-shell notranslate">

<div class="highlight">

    export DEVELOPER_DIR="/path/to/Xcode.app/Contents/Developer/"

</div>

</div>

Further, you can use the following command to find out which macOS SDK
will be used

<div class="highlight-shell notranslate">

<div class="highlight">

    xcrun -sdk macosx --show-sdk-version

</div>

</div>

</div>

<div id="binary-size-minimization" class="section">

#### Binary Size Minimization<a
href="https://ml-explore.github.io/mlx/build/html/#binary-size-minimization"
class="headerlink" title="Link to this heading">#</a>

To produce a smaller binary use the CMake flags
<span class="pre">`CMAKE_BUILD_TYPE=MinSizeRel`</span> and
<span class="pre">`BUILD_SHARED_LIBS=ON`</span>.

The MLX CMake build has several additional options to make smaller
binaries. For example, if you don’t need the CPU backend or support for
safetensors and GGUF, you can do:

<div class="highlight-shell notranslate">

<div class="highlight">

    cmake .. \
      -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DBUILD_SHARED_LIBS=ON \
      -DMLX_BUILD_CPU=OFF \
      -DMLX_BUILD_SAFETENSORS=OFF \
      -DMLX_BUILD_GGUF=OFF \
      -DMLX_METAL_JIT=ON

</div>

</div>

THE <span class="pre">`MLX_METAL_JIT`</span> flag minimizes the size of
the MLX Metal library which contains pre-built GPU kernels. This
substantially reduces the size of the Metal library by run-time
compiling kernels the first time they are used in MLX on a given
machine. Note run-time compilation incurs a cold-start cost which can be
anwywhere from a few hundred millisecond to a few seconds depending on
the application. Once a kernel is compiled, it will be cached by the
system. The Metal kernel cache persists across reboots.

</div>

</div>

<div id="id3" class="section">

### Troubleshooting<a href="https://ml-explore.github.io/mlx/build/html/#id3"
class="headerlink" title="Link to this heading">#</a>

<div id="metal-not-found" class="section">

#### Metal not found<a href="https://ml-explore.github.io/mlx/build/html/#metal-not-found"
class="headerlink" title="Link to this heading">#</a>

You see the following error when you try to build:

<div class="highlight-shell notranslate">

<div class="highlight">

    error: unable to find utility "metal", not a developer tool or in PATH

</div>

</div>

To fix this, first make sure you have Xcode installed:

<div class="highlight-shell notranslate">

<div class="highlight">

    xcode-select --install

</div>

</div>

Then set the active developer directory:

<div class="highlight-shell notranslate">

<div class="highlight">

    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

</div>

</div>

</div>

<div id="x86-shell" class="section">

#### x86 Shell<a href="https://ml-explore.github.io/mlx/build/html/#x86-shell"
class="headerlink" title="Link to this heading">#</a>

If the output of
<span class="pre">`uname`</span>` `<span class="pre">`-p`</span> is
<span class="pre">`x86`</span> then your shell is running as x86 via
Rosetta instead of natively.

To fix this, find the application in Finder
(<span class="pre">`/Applications`</span> for iTerm,
<span class="pre">`/Applications/Utilities`</span> for Terminal),
right-click, and click “Get Info”. Uncheck “Open using Rosetta”, close
the “Get Info” window, and restart your terminal.

Verify the terminal is now running natively the following command:

<div class="highlight-shell notranslate">

<div class="highlight">

    $ uname -p
    arm

</div>

</div>

Also check that cmake is using the correct architecture:

<div class="highlight-shell notranslate">

<div class="highlight">

    $ cmake --system-information | grep CMAKE_HOST_SYSTEM_PROCESSOR
    CMAKE_HOST_SYSTEM_PROCESSOR "arm64"

</div>

</div>

If you see <span class="pre">`"x86_64"`</span>, try re-installing
<span class="pre">`cmake`</span>. If you see
<span class="pre">`"arm64"`</span> but the build errors out with
“Building for x86_64 on macOS is not supported.” wipe your build cache
with
<span class="pre">`rm`</span>` `<span class="pre">`-rf`</span>` `<span class="pre">`build/`</span>
and try again.

</div>

</div>

</div>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/index.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

MLX

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/quick_start.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Quick Start Guide

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
  href="https://ml-explore.github.io/mlx/build/html/#python-installation"
  class="reference internal nav-link">Python Installation</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#troubleshooting"
    class="reference internal nav-link">Troubleshooting</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#build-from-source"
  class="reference internal nav-link">Build from source</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#build-requirements"
    class="reference internal nav-link">Build Requirements</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#python-api"
    class="reference internal nav-link">Python API</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#c-api"
    class="reference internal nav-link">C++ API</a>
    - <a
      href="https://ml-explore.github.io/mlx/build/html/#binary-size-minimization"
      class="reference internal nav-link">Binary Size Minimization</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#id3"
    class="reference internal nav-link">Troubleshooting</a>
    - <a href="https://ml-explore.github.io/mlx/build/html/#metal-not-found"
      class="reference internal nav-link">Metal not found</a>
    - <a href="https://ml-explore.github.io/mlx/build/html/#x86-shell"
      class="reference internal nav-link">x86 Shell</a>

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
