Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (launching_distributed.md):
- Launcher usage doc for multi-host runs with hostfiles and backends.
- Add quick sanity checks and environment variable hints.
-->

## Curated Notes

- Sanity check local run first: `mlx.launch -n 2 your_script.py`.
- Ensure passwordless SSH between hosts; sync Python env and script paths across machines.
- Set env vars per rank when needed: `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` (some backends infer these automatically).


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/launching_distributed.rst"
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

# Launching Distributed Programs

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#usage"
  class="reference internal nav-link">Usage</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#providing-hosts"
    class="reference internal nav-link">Providing Hosts</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#setting-up-remote-hosts"
    class="reference internal nav-link">Setting up Remote Hosts</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#mpi-specifics"
  class="reference internal nav-link">MPI Specifics</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#ring-specifics"
  class="reference internal nav-link">Ring Specifics</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="launching-distributed-programs" class="section">

<span id="usage-launch-distributed"></span>

# Launching Distributed Programs<a
href="https://ml-explore.github.io/mlx/build/html/#launching-distributed-programs"
class="headerlink" title="Link to this heading">#</a>

Installing the MLX python package provides a helper script
<span class="pre">`mlx.launch`</span> that can be used to run python
scripts distributed on several nodes. It allows launching using either
the MPI backend or the ring backend. See the <a
href="https://ml-explore.github.io/mlx/build/html/usage/distributed.html"
class="reference internal"><span class="doc">distributed docs</span></a>
for the different backends.

<div id="usage" class="section">

## Usage<a href="https://ml-explore.github.io/mlx/build/html/#usage"
class="headerlink" title="Link to this heading">#</a>

The minimal usage example of <span class="pre">`mlx.launch`</span> is
simply

<div class="highlight-shell notranslate">

<div class="highlight">

    mlx.launch --hosts ip1,ip2 my_script.py

</div>

</div>

or for testing on localhost

<div class="highlight-shell notranslate">

<div class="highlight">

    mlx.launch -n 2 my_script.py

</div>

</div>

The <span class="pre">`mlx.launch`</span> command connects to the
provided host and launches the input script on each host. It monitors
each of the launched processes and terminates the rest if one of them
fails unexpectedly or if <span class="pre">`mlx.launch`</span> is
terminated. It also takes care of forwarding the output of each remote
process to stdout and stderr respectively.

<div id="providing-hosts" class="section">

### Providing Hosts<a href="https://ml-explore.github.io/mlx/build/html/#providing-hosts"
class="headerlink" title="Link to this heading">#</a>

Hosts can be provided as command line arguments, like above, but the way
that allows to fully define a list of hosts is via a JSON hostfile. The
hostfile has a very simple schema. It is simply a list of objects that
define each host via a hostname to ssh to and a list of IPs to utilize
for the communication.

<div class="highlight-json notranslate">

<div class="highlight">

    [
        {"ssh": "hostname1", "ips": ["123.123.1.1", "123.123.2.1"]},
        {"ssh": "hostname2", "ips": ["123.123.1.2", "123.123.2.2"]}
    ]

</div>

</div>

You can use
<span class="pre">`mlx.distributed_config`</span>` `<span class="pre">`--over`</span>` `<span class="pre">`ethernet`</span>
to create a hostfile with IPs corresponding to the
<span class="pre">`en0`</span> interface.

</div>

<div id="setting-up-remote-hosts" class="section">

### Setting up Remote Hosts<a
href="https://ml-explore.github.io/mlx/build/html/#setting-up-remote-hosts"
class="headerlink" title="Link to this heading">#</a>

In order to be able to launch the script on each host we need to be able
to connect via ssh. Moreover the input script and python binary need to
be on each host and on the same path. A good checklist to debug errors
is the following:

- <span class="pre">`ssh`</span>` `<span class="pre">`hostname`</span>
  works without asking for password or host confirmation

- the python binary is available on all hosts at the same path. You can
  use
  <span class="pre">`mlx.launch`</span>` `<span class="pre">`--print-python`</span>
  to see what that path is.

- the script you want to run is available on all hosts at the same path

</div>

</div>

<div id="mpi-specifics" class="section">

<span id="id1"></span>

## MPI Specifics<a href="https://ml-explore.github.io/mlx/build/html/#mpi-specifics"
class="headerlink" title="Link to this heading">#</a>

One can use MPI by passing
<span class="pre">`--backend`</span>` `<span class="pre">`mpi`</span> to
<span class="pre">`mlx.launch`</span>. In that case,
<span class="pre">`mlx.launch`</span> is a thin wrapper over
<span class="pre">`mpirun`</span>. Moreover,

- The IPs in the hostfile are ignored

- The ssh connectivity requirement is stronger as every node needs to be
  able to connect to every other node

- <span class="pre">`mpirun`</span> needs to be available on every node
  at the same path

Finally, one can pass arguments to <span class="pre">`mpirun`</span>
using <span class="pre">`--mpi-arg`</span>. For instance to choose a
specific interface for the byte-transfer-layer of MPI we can call
<span class="pre">`mlx.launch`</span> as follows:

<div class="highlight-shell notranslate">

<div class="highlight">

    mlx.launch --backend mpi --mpi-arg '--mca btl_tcp_if_include en0' --hostfile hosts.json my_script.py

</div>

</div>

</div>

<div id="ring-specifics" class="section">

<span id="id2"></span>

## Ring Specifics<a href="https://ml-explore.github.io/mlx/build/html/#ring-specifics"
class="headerlink" title="Link to this heading">#</a>

The ring backend, which is also the default backend, can be explicitly
selected with the argument
<span class="pre">`--backend`</span>` `<span class="pre">`ring`</span>.
The ring backend has some specific requirements and arguments that are
different to MPI:

- The argument <span class="pre">`--hosts`</span> only accepts IPs and
  not hostnames. If we need to ssh to a hostname that does not
  correspond to the IP we want to bind to we have to provide a hostfile.

- <span class="pre">`--starting-port`</span> defines the port to bind to
  on the remote hosts. Specifically rank 0 for the first IP will use
  this port and each subsequent IP or rank will add 1 to this port.

- <span class="pre">`--connections-per-ip`</span> allows us to increase
  the number of connections between neighboring nodes. This corresponds
  to
  <span class="pre">`--mca`</span>` `<span class="pre">`btl_tcp_links`</span>` `<span class="pre">`2`</span>
  for <span class="pre">`mpirun`</span>.

</div>

</div>

<div class="prev-next-area">

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#usage"
  class="reference internal nav-link">Usage</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#providing-hosts"
    class="reference internal nav-link">Providing Hosts</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#setting-up-remote-hosts"
    class="reference internal nav-link">Setting up Remote Hosts</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#mpi-specifics"
  class="reference internal nav-link">MPI Specifics</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#ring-specifics"
  class="reference internal nav-link">Ring Specifics</a>

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
