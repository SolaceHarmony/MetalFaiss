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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/random.rst"
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

# Random

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="random" class="section">

<span id="id1"></span>

# Random<a href="https://ml-explore.github.io/mlx/build/html/#random"
class="headerlink" title="Link to this heading">#</a>

Random sampling functions in MLX use an implicit global PRNG state by
default. However, all function take an optional
<span class="pre">`key`</span> keyword argument for when more
fine-grained control or explicit state management is needed.

For example, you can generate random numbers with:

<div class="highlight-python notranslate">

<div class="highlight">

    for _ in range(3):
      print(mx.random.uniform())

</div>

</div>

which will print a sequence of unique pseudo random numbers.
Alternatively you can explicitly set the key:

<div class="highlight-python notranslate">

<div class="highlight">

    key = mx.random.key(0)
    for _ in range(3):
      print(mx.random.uniform(key=key))

</div>

</div>

which will yield the same pseudo random number at each iteration.

Following
<a href="https://jax.readthedocs.io/en/latest/jep/263-prng.html"
class="reference external">JAX’s PRNG design</a> we use a splittable
version of Threefry, which is a counter-based PRNG.

## Curated Notes

- Reproducibility:
  - Global: `mx.random.seed(n)` seeds the global generator.
  - Local: pass `key=mx.random.key(n)` for explicit, deterministic draws. Reuse the same key to reproduce the same value; use `mx.random.split(key, num=...)` to create independent subkeys.
- Device control: random ops do not take a `device=` argument; use default device or per‑op `stream`.
- Dtypes/shapes: most random APIs accept `shape=` and `dtype=`; if omitted, defaults are used (often float32). Construct then `astype(...)` if an API lacks a `dtype` parameter.
- Ranges/semantics:
  - `randint(low, high, ...)` is typically half‑open [low, high).
  - `permutation(x, axis=...)` shuffles along an axis or returns a permuted copy for 1‑D inputs.
  - `categorical(logits, axis=...)` samples indices; ensure logits axis matches `axis`.

### Reproducible Patterns

```python
import mlx.core as mx

# Global seed
mx.random.seed(42)
a = mx.random.normal(shape=(2, 3))

# Explicit keys
key = mx.random.key(0)
u1 = mx.random.uniform(shape=(2, 2), key=key)
u2 = mx.random.uniform(shape=(2, 2), key=key)   # identical to u1

# Independent subkeys
k1, k2 = mx.random.split(mx.random.key(123), num=2)
n1 = mx.random.normal(shape=(4,), key=k1)
n2 = mx.random.normal(shape=(4,), key=k2)
```

### Common Distributions

```python
# Uniform in [low, high)
u = mx.random.uniform(low=-1.0, high=1.0, shape=(3, 3))

# Normal with loc/scale
n = mx.random.normal(loc=0.0, scale=2.0, shape=(1024,))

# Truncated normal in [lower, upper]
tn = mx.random.truncated_normal(lower=-2.0, upper=2.0, shape=(5,))

# Categorical over last axis by default
logits = mx.random.normal(shape=(8, 10))
idx = mx.random.categorical(logits, axis=-1)

# Integers: [low, high)
ri = mx.random.randint(low=0, high=10, shape=(4, 4), dtype=mx.int32)

# Permutation
p = mx.random.permutation(mx.arange(10))
```

### Device/Stream Examples

```python
import mlx.core as mx

# Route random draws to CPU explicitly
cpu_u = mx.random.uniform(shape=(2, 2), stream=mx.cpu)

# Or set global default device (no device= on ops)
mx.set_default_device(mx.cpu)
cpu_n = mx.random.normal(shape=(128,))
```

### Vectorizing With Independent Randomness

When using `mx.vmap`, split a single seed key into per‑example subkeys so each parallel instance has its own independent stream.

```python
import mlx.core as mx

def my_random_function(key):
    # Generates a 5x5 normal draw using the provided key
    return mx.random.normal(shape=(5, 5), key=key)

# 1) Start from a reproducible key
initial_key = mx.random.key(42)

# 2) Split into subkeys for a batch
batch_size = 4
subkeys = mx.random.split(initial_key, num=batch_size)

# 3) Vectorize: vmap maps over the first axis of subkeys
vf = mx.vmap(my_random_function)
parallel = vf(subkeys)  # shape: (4, 5, 5)

# Reproducibility: splitting again from the same initial key yields the same draws
subkeys2 = mx.random.split(initial_key, num=batch_size)
parallel2 = vf(subkeys2)
print(mx.allclose(parallel, parallel2).item())  # True

## MLX vs PyTorch Random (Quick Comparison)

- State model:
  - MLX: explicit, functional keys; pass `key=` to each op. Split keys with `mx.random.split` for independent streams.
  - PyTorch: implicit global state (`torch.manual_seed`); finer control via stateful `torch.Generator`.
- Reproducibility:
  - MLX: seed → key → split; deterministic across `vmap`/parallel code when subkeys are managed explicitly.
  - PyTorch: global/manual seeding works in simple cases; care needed across threads/devices.
- APIs:
  - MLX `normal(shape, loc, scale, key=...)`; PyTorch `randn(size) * std + mean`.
  - MLX `uniform(low, high, shape, key=...)`; PyTorch `rand(size) * (high-low) + low`.
  - MLX `randint(low, high, shape, key=...)` (half‑open); PyTorch `randint(low, high, size)` similar.
  - MLX `permutation(x, axis, key=...)`; PyTorch `randperm(n)` returns indices for 1‑D.

Example (MNIST‑style batch):
```python
# MLX
k = mx.random.key(0)
k_img, k_lbl = mx.random.split(k, num=2)
noise = mx.random.normal(shape=(64, 1, 28, 28), key=k_img)
labels = mx.random.randint(0, 10, shape=(64,), key=k_lbl)

# PyTorch (conceptually)
# torch.manual_seed(0)
# noise = torch.randn(64, 1, 28, 28)
# labels = torch.randint(0, 10, (64,))
```
 
### Avoid Seeding Inside Transformed Code

- Don’t call `mx.random.seed(...)` inside functions you pass to `mx.vmap`, `mx.value_and_grad`, or `mx.compile`.
- Seeding mutates global state and can collapse independence across mapped/parallel instances or break caching.

Bad (global seed inside mapped fn):
```python
def f_bad(x):
    mx.random.seed(0)                 # resets global state every call
    return mx.random.normal(shape=x.shape)
vf = mx.vmap(f_bad)
vf(mx.ones((4, 8)))                   # rows likely identical
```

Good (explicit subkeys):
```python
def f_good(key, x):
    return mx.random.normal(shape=x.shape, key=key)
vf = mx.vmap(f_good)
keys = mx.random.split(mx.random.key(0), num=4)
vf(keys, mx.ones((4, 8)))            # independent rows
```

### Bridging Missing Distributions (Patterns)

Exponential via inverse CDF:
```python
def exponential(shape, scale=1.0, key=None, dtype=mx.float32):
    u = mx.random.uniform(shape=shape, key=key).astype(dtype)
    eps = mx.array(1e-9, dtype=dtype)
    return -mx.array(scale, dtype=dtype) * mx.log(mx.maximum(1.0 - u, eps))
```

Poisson via Knuth for small λ and Normal approximation for large λ:
```python
def poisson(shape, lam=1.0, key=None):
    lam = float(lam)
    if lam <= 0: return mx.zeros(shape, dtype=mx.int32)
    if lam > 15:
        n = mx.random.normal(shape=shape, loc=lam, scale=mx.sqrt(lam), key=key)
        return mx.maximum(mx.round(n), 0).astype(mx.int32)
    # Knuth
    L = mx.exp(-lam)
    k = mx.zeros(shape, dtype=mx.int32)
    p = mx.ones(shape)
    for _ in range(max(100, int(lam*5))):
        cont = p >= L
        if not mx.any(cont): break
        k = mx.where(cont, k + 1, k)
        u = mx.random.uniform(shape=shape, key=key)
        p = mx.where(cont, p * u, p)
    return k
```

Integer uniform (if `randint` is unavailable in your MLX version):
```python
def randint(low, high, shape, key=None, dtype=mx.int32):
    u = mx.random.uniform(shape=shape, low=float(low), high=float(high), key=key)
    return mx.floor(u).astype(dtype)
```

### Shuffling and Permutations

- 1‑D: `mx.random.permutation(n, key=...)` returns indices 0..n‑1 permuted.
- Array: permute along an axis via advanced indexing.

```python
def shuffle_first_dim(x, key=None):
    idx = mx.random.permutation(x.shape[0], key=key)
    return x[idx]
```

Epoch/worker‑safe shuffles:
```python
def epoch_key(base_seed, epoch, worker=0):
    return mx.random.key(int(base_seed) ^ (epoch * 0x9E3779B97F4A7C15) ^ worker)

k = epoch_key(0, epoch=5, worker=rank)
idx = mx.random.permutation(N, key=k)
```

## PyTorch/NumPy‑Style RNG Shim (Compatibility)

If you want a front‑end that feels like `torch.Generator`/global NumPy state without exposing keys to call sites, wrap MLX RNG in a tiny class. Internally it derives unique keys per call so parallel code stays reproducible, but the API looks familiar.

```python
import mlx.core as mx

class RNG:
    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._ctr = 0

    def _next_key(self):
        # Derive a fresh key per call; no global state
        self._ctr += 1
        return mx.random.key(self.seed ^ (0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFF) ^ self._ctr)

    # Torch/NumPy‑like methods (no key arg exposed)
    def normal(self, shape, loc=0.0, scale=1.0, dtype=mx.float32):
        return mx.random.normal(shape=shape, loc=loc, scale=scale, key=self._next_key()).astype(dtype)

    def uniform(self, shape, low=0.0, high=1.0, dtype=mx.float32):
        return mx.random.uniform(shape=shape, low=low, high=high, key=self._next_key()).astype(dtype)

    def randint(self, low, high, shape, dtype=mx.int32):
        return mx.random.randint(low=low, high=high, shape=shape, dtype=dtype, key=self._next_key())

    def permutation(self, n_or_x):
        return mx.random.permutation(n_or_x, key=self._next_key())

    def categorical(self, logits, num_samples):
        return mx.random.categorical(logits, num_samples=num_samples, key=self._next_key())

# Usage
g = RNG(seed=123)
x = g.normal((2, 3))
idx = g.permutation(10)
```

Notes:
- Callers don’t pass keys; you get deterministic, independent draws per method call.
- For strict global‑state emulation, expose `set_seed/get_seed` that mutate a module‑level RNG instance; prefer the instance API above for clarity and testability.

## Factory Helpers (rand/randn/like)

MLX doesn’t ship NumPy/Torch‑style factory shorthands, but they’re easy to add:

```python
import mlx.core as mx

def rand(shape, low=0.0, high=1.0, dtype=mx.float32, key=None):
    return mx.random.uniform(shape=shape, low=low, high=high, key=key).astype(dtype)

def randn(shape, mean=0.0, std=1.0, dtype=mx.float32, key=None):
    return mx.random.normal(shape=shape, loc=mean, scale=std, key=key).astype(dtype)

def randint(shape, low, high, dtype=mx.int32, key=None):
    return mx.random.randint(low=low, high=high, shape=shape, dtype=dtype, key=key)

def rand_like(x, low=0.0, high=1.0, key=None):
    return mx.random.uniform(shape=x.shape, low=low, high=high, key=key).astype(x.dtype)

def randn_like(x, mean=0.0, std=1.0, key=None):
    return mx.random.normal(shape=x.shape, loc=mean, scale=std, key=key).astype(x.dtype)

def randint_like(x, low, high, key=None):
    # Use x.dtype if it is an integer type; else default to int32
    int_dtype = x.dtype if str(x.dtype).startswith('int') else mx.int32
    return mx.random.randint(low=low, high=high, shape=x.shape, dtype=int_dtype, key=key)
```

These mirror NumPy/Torch ergonomics and keep dtype/device consistent with a reference array when using the “_like” variants.
```

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.bernoulli.html#mlx.core.random.bernoulli"
class="reference internal" title="mlx.core.random.bernoulli"><span
class="pre"><code class="sourceCode python">bernoulli</code></span></a>(\[p, shape, key, stream\]) | Generate Bernoulli random values. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html#mlx.core.random.categorical"
class="reference internal" title="mlx.core.random.categorical"><span
class="pre"><code
class="sourceCode python">categorical</code></span></a>(logits\[, axis, shape, ...\]) | Sample from a categorical distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.gumbel.html#mlx.core.random.gumbel"
class="reference internal" title="mlx.core.random.gumbel"><span
class="pre"><code class="sourceCode python">gumbel</code></span></a>(\[shape, dtype, key, stream\]) | Sample from the standard Gumbel distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.key.html#mlx.core.random.key"
class="reference internal" title="mlx.core.random.key"><span
class="pre"><code class="sourceCode python">key</code></span></a>(seed) | Get a PRNG key from a seed. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html#mlx.core.random.normal"
class="reference internal" title="mlx.core.random.normal"><span
class="pre"><code class="sourceCode python">normal</code></span></a>(\[shape, dtype, loc, scale, key, stream\]) | Generate normally distributed random numbers. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.multivariate_normal.html#mlx.core.random.multivariate_normal"
class="reference internal"
title="mlx.core.random.multivariate_normal"><span class="pre"><code
class="sourceCode python">multivariate_normal</code></span></a>(mean, cov\[, shape, ...\]) | Generate jointly-normal random samples given a mean and covariance. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.randint.html#mlx.core.random.randint"
class="reference internal" title="mlx.core.random.randint"><span
class="pre"><code class="sourceCode python">randint</code></span></a>(low, high\[, shape, dtype, key, stream\]) | Generate random integers from the given interval. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.seed.html#mlx.core.random.seed"
class="reference internal" title="mlx.core.random.seed"><span
class="pre"><code class="sourceCode python">seed</code></span></a>(seed) | Seed the global PRNG. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.split.html#mlx.core.random.split"
class="reference internal" title="mlx.core.random.split"><span
class="pre"><code class="sourceCode python">split</code></span></a>(key\[, num, stream\]) | Split a PRNG key into sub keys. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.truncated_normal.html#mlx.core.random.truncated_normal"
class="reference internal"
title="mlx.core.random.truncated_normal"><span class="pre"><code
class="sourceCode python">truncated_normal</code></span></a>(lower, upper\[, shape, ...\]) | Generate values from a truncated normal distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.uniform.html#mlx.core.random.uniform"
class="reference internal" title="mlx.core.random.uniform"><span
class="pre"><code class="sourceCode python">uniform</code></span></a>(\[low, high, shape, dtype, key, stream\]) | Generate uniformly distributed random numbers. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.laplace.html#mlx.core.random.laplace"
class="reference internal" title="mlx.core.random.laplace"><span
class="pre"><code class="sourceCode python">laplace</code></span></a>(\[shape, dtype, loc, scale, key, stream\]) | Sample numbers from a Laplace distribution. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.permutation.html#mlx.core.random.permutation"
class="reference internal" title="mlx.core.random.permutation"><span
class="pre"><code
class="sourceCode python">permutation</code></span></a>(x\[, axis, key, stream\]) | Generate a random permutation or permute the entries of an array. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.zeros_like.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.zeros_like

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.bernoulli.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.random.bernoulli

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
