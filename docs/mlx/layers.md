Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (layers.md):
- Large index of nn layers; newcomers stumble on shape conventions and pooling/flatten math.
-->

## Curated Notes

- Input convention: most layers expect `(N, C, ...)` (e.g., Conv2d: `(N, C, H, W)`).
- After spatial layers, `nn.Flatten()` helps build MLP heads: verify `C*H*W` matches the next `Linear`.
- For depthwise conv: `groups=in_channels` with `out_channels=in_channels`.

### Examples

```python
import mlx.core as mx
import mlx.nn as nn

# NCHW input
x = mx.random.normal((8, 3, 32, 32))
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
y = conv(x)                         # (8, 16, 32, 32)

pool = nn.AvgPool2d(2)
p = pool(y)                         # (8, 16, 16, 16)

flat = nn.Flatten()
f = flat(p)                         # (8, 16*16*16)
head = nn.Linear(16*16*16, 10)
logits = head(f)                    # (8, 10)

# Depthwise conv (per-channel)
dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
z = dw(y)
```


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/layers.rst"
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

# Layers

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="layers" class="section">

<span id="id1"></span>

# Layers<a href="https://ml-explore.github.io/mlx/build/html/#layers"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ALiBi.html#mlx.nn.ALiBi"
class="reference internal" title="mlx.nn.ALiBi"><span class="pre"><code
class="sourceCode python">ALiBi</code></span></a>() |  |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool1d.html#mlx.nn.AvgPool1d"
class="reference internal" title="mlx.nn.AvgPool1d"><span
class="pre"><code class="sourceCode python">AvgPool1d</code></span></a>(kernel_size\[, stride, padding\]) | Applies 1-dimensional average pooling. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html#mlx.nn.AvgPool2d"
class="reference internal" title="mlx.nn.AvgPool2d"><span
class="pre"><code class="sourceCode python">AvgPool2d</code></span></a>(kernel_size\[, stride, padding\]) | Applies 2-dimensional average pooling. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool3d.html#mlx.nn.AvgPool3d"
class="reference internal" title="mlx.nn.AvgPool3d"><span
class="pre"><code class="sourceCode python">AvgPool3d</code></span></a>(kernel_size\[, stride, padding\]) | Applies 3-dimensional average pooling. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.BatchNorm.html#mlx.nn.BatchNorm"
class="reference internal" title="mlx.nn.BatchNorm"><span
class="pre"><code class="sourceCode python">BatchNorm</code></span></a>(num_features\[, eps, momentum, ...\]) | Applies Batch Normalization over a 2D or 3D input. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.CELU.html#mlx.nn.CELU"
class="reference internal" title="mlx.nn.CELU"><span class="pre"><code
class="sourceCode python">CELU</code></span></a>(\[alpha\]) | Applies the Continuously Differentiable Exponential Linear Unit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv1d.html#mlx.nn.Conv1d"
class="reference internal" title="mlx.nn.Conv1d"><span class="pre"><code
class="sourceCode python">Conv1d</code></span></a>(in_channels, out_channels, kernel_size) | Applies a 1-dimensional convolution over the multi-channel input sequence. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html#mlx.nn.Conv2d"
class="reference internal" title="mlx.nn.Conv2d"><span class="pre"><code
class="sourceCode python">Conv2d</code></span></a>(in_channels, out_channels, kernel_size) | Applies a 2-dimensional convolution over the multi-channel input image. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv3d.html#mlx.nn.Conv3d"
class="reference internal" title="mlx.nn.Conv3d"><span class="pre"><code
class="sourceCode python">Conv3d</code></span></a>(in_channels, out_channels, kernel_size) | Applies a 3-dimensional convolution over the multi-channel input image. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html#mlx.nn.ConvTranspose1d"
class="reference internal" title="mlx.nn.ConvTranspose1d"><span
class="pre"><code
class="sourceCode python">ConvTranspose1d</code></span></a>(in_channels, out_channels, ...) | Applies a 1-dimensional transposed convolution over the multi-channel input sequence. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose2d.html#mlx.nn.ConvTranspose2d"
class="reference internal" title="mlx.nn.ConvTranspose2d"><span
class="pre"><code
class="sourceCode python">ConvTranspose2d</code></span></a>(in_channels, out_channels, ...) | Applies a 2-dimensional transposed convolution over the multi-channel input image. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose3d.html#mlx.nn.ConvTranspose3d"
class="reference internal" title="mlx.nn.ConvTranspose3d"><span
class="pre"><code
class="sourceCode python">ConvTranspose3d</code></span></a>(in_channels, out_channels, ...) | Applies a 3-dimensional transposed convolution over the multi-channel input image. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout.html#mlx.nn.Dropout"
class="reference internal" title="mlx.nn.Dropout"><span
class="pre"><code class="sourceCode python">Dropout</code></span></a>(\[p\]) | Randomly zero a portion of the elements during training. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout2d.html#mlx.nn.Dropout2d"
class="reference internal" title="mlx.nn.Dropout2d"><span
class="pre"><code class="sourceCode python">Dropout2d</code></span></a>(\[p\]) | Apply 2D channel-wise dropout during training. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout3d.html#mlx.nn.Dropout3d"
class="reference internal" title="mlx.nn.Dropout3d"><span
class="pre"><code class="sourceCode python">Dropout3d</code></span></a>(\[p\]) | Apply 3D channel-wise dropout during training. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding"
class="reference internal" title="mlx.nn.Embedding"><span
class="pre"><code class="sourceCode python">Embedding</code></span></a>(num_embeddings, dims) | Implements a simple lookup table that maps each input integer to a high-dimensional vector. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ELU.html#mlx.nn.ELU"
class="reference internal" title="mlx.nn.ELU"><span class="pre"><code
class="sourceCode python">ELU</code></span></a>(\[alpha\]) | Applies the Exponential Linear Unit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GELU.html#mlx.nn.GELU"
class="reference internal" title="mlx.nn.GELU"><span class="pre"><code
class="sourceCode python">GELU</code></span></a>(\[approx\]) | Applies the Gaussian Error Linear Units. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GLU.html#mlx.nn.GLU"
class="reference internal" title="mlx.nn.GLU"><span class="pre"><code
class="sourceCode python">GLU</code></span></a>(\[axis\]) | Applies the gated linear unit function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GroupNorm.html#mlx.nn.GroupNorm"
class="reference internal" title="mlx.nn.GroupNorm"><span
class="pre"><code class="sourceCode python">GroupNorm</code></span></a>(num_groups, dims\[, eps, affine, ...\]) | Applies Group Normalization \[1\] to the inputs. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GRU.html#mlx.nn.GRU"
class="reference internal" title="mlx.nn.GRU"><span class="pre"><code
class="sourceCode python">GRU</code></span></a>(input_size, hidden_size\[, bias\]) | A gated recurrent unit (GRU) RNN layer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardShrink.html#mlx.nn.HardShrink"
class="reference internal" title="mlx.nn.HardShrink"><span
class="pre"><code class="sourceCode python">HardShrink</code></span></a>() | Applies the HardShrink function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardTanh.html#mlx.nn.HardTanh"
class="reference internal" title="mlx.nn.HardTanh"><span
class="pre"><code class="sourceCode python">HardTanh</code></span></a>() | Applies the HardTanh function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Hardswish.html#mlx.nn.Hardswish"
class="reference internal" title="mlx.nn.Hardswish"><span
class="pre"><code class="sourceCode python">Hardswish</code></span></a>() | Applies the hardswish function, element-wise. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.InstanceNorm.html#mlx.nn.InstanceNorm"
class="reference internal" title="mlx.nn.InstanceNorm"><span
class="pre"><code
class="sourceCode python">InstanceNorm</code></span></a>(dims\[, eps, affine\]) | Applies instance normalization \[1\] on the inputs. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LayerNorm.html#mlx.nn.LayerNorm"
class="reference internal" title="mlx.nn.LayerNorm"><span
class="pre"><code class="sourceCode python">LayerNorm</code></span></a>(dims\[, eps, affine, bias\]) | Applies layer normalization \[1\] on the inputs. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LeakyReLU.html#mlx.nn.LeakyReLU"
class="reference internal" title="mlx.nn.LeakyReLU"><span
class="pre"><code class="sourceCode python">LeakyReLU</code></span></a>(\[negative_slope\]) | Applies the Leaky Rectified Linear Unit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear"
class="reference internal" title="mlx.nn.Linear"><span class="pre"><code
class="sourceCode python">Linear</code></span></a>(input_dims, output_dims\[, bias\]) | Applies an affine transformation to the input. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSigmoid.html#mlx.nn.LogSigmoid"
class="reference internal" title="mlx.nn.LogSigmoid"><span
class="pre"><code class="sourceCode python">LogSigmoid</code></span></a>() | Applies the Log Sigmoid function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSoftmax.html#mlx.nn.LogSoftmax"
class="reference internal" title="mlx.nn.LogSoftmax"><span
class="pre"><code class="sourceCode python">LogSoftmax</code></span></a>() | Applies the Log Softmax function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LSTM.html#mlx.nn.LSTM"
class="reference internal" title="mlx.nn.LSTM"><span class="pre"><code
class="sourceCode python">LSTM</code></span></a>(input_size, hidden_size\[, bias\]) | An LSTM recurrent layer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool1d.html#mlx.nn.MaxPool1d"
class="reference internal" title="mlx.nn.MaxPool1d"><span
class="pre"><code class="sourceCode python">MaxPool1d</code></span></a>(kernel_size\[, stride, padding\]) | Applies 1-dimensional max pooling. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool2d.html#mlx.nn.MaxPool2d"
class="reference internal" title="mlx.nn.MaxPool2d"><span
class="pre"><code class="sourceCode python">MaxPool2d</code></span></a>(kernel_size\[, stride, padding\]) | Applies 2-dimensional max pooling. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool3d.html#mlx.nn.MaxPool3d"
class="reference internal" title="mlx.nn.MaxPool3d"><span
class="pre"><code class="sourceCode python">MaxPool3d</code></span></a>(kernel_size\[, stride, padding\]) | Applies 3-dimensional max pooling. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Mish.html#mlx.nn.Mish"
class="reference internal" title="mlx.nn.Mish"><span class="pre"><code
class="sourceCode python">Mish</code></span></a>() | Applies the Mish function, element-wise. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html#mlx.nn.MultiHeadAttention"
class="reference internal" title="mlx.nn.MultiHeadAttention"><span
class="pre"><code
class="sourceCode python">MultiHeadAttention</code></span></a>(dims, num_heads\[, ...\]) | Implements the scaled dot product attention with multiple heads. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.PReLU.html#mlx.nn.PReLU"
class="reference internal" title="mlx.nn.PReLU"><span class="pre"><code
class="sourceCode python">PReLU</code></span></a>(\[num_parameters, init\]) | Applies the element-wise parametric ReLU. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedEmbedding.html#mlx.nn.QuantizedEmbedding"
class="reference internal" title="mlx.nn.QuantizedEmbedding"><span
class="pre"><code
class="sourceCode python">QuantizedEmbedding</code></span></a>(num_embeddings, dims\[, ...\]) | The same as <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding"
class="reference internal" title="mlx.nn.Embedding"><span
class="pre"><code class="sourceCode python">Embedding</code></span></a> but with a quantized weight matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html#mlx.nn.QuantizedLinear"
class="reference internal" title="mlx.nn.QuantizedLinear"><span
class="pre"><code
class="sourceCode python">QuantizedLinear</code></span></a>(input_dims, output_dims\[, ...\]) | Applies an affine transformation to the input using a quantized weight matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RMSNorm.html#mlx.nn.RMSNorm"
class="reference internal" title="mlx.nn.RMSNorm"><span
class="pre"><code class="sourceCode python">RMSNorm</code></span></a>(dims\[, eps\]) | Applies Root Mean Square normalization \[1\] to the inputs. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ReLU.html#mlx.nn.ReLU"
class="reference internal" title="mlx.nn.ReLU"><span class="pre"><code
class="sourceCode python">ReLU</code></span></a>() | Applies the Rectified Linear Unit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ReLU6.html#mlx.nn.ReLU6"
class="reference internal" title="mlx.nn.ReLU6"><span class="pre"><code
class="sourceCode python">ReLU6</code></span></a>() | Applies the Rectified Linear Unit 6. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RNN.html#mlx.nn.RNN"
class="reference internal" title="mlx.nn.RNN"><span class="pre"><code
class="sourceCode python">RNN</code></span></a>(input_size, hidden_size\[, bias, ...\]) | An Elman recurrent layer. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html#mlx.nn.RoPE"
class="reference internal" title="mlx.nn.RoPE"><span class="pre"><code
class="sourceCode python">RoPE</code></span></a>(dims\[, traditional, base, scale\]) | Implements the rotary positional encoding. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SELU.html#mlx.nn.SELU"
class="reference internal" title="mlx.nn.SELU"><span class="pre"><code
class="sourceCode python">SELU</code></span></a>() | Applies the Scaled Exponential Linear Unit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Sequential.html#mlx.nn.Sequential"
class="reference internal" title="mlx.nn.Sequential"><span
class="pre"><code class="sourceCode python">Sequential</code></span></a>(\*modules) | A layer that calls the passed callables in order. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Sigmoid.html#mlx.nn.Sigmoid"
class="reference internal" title="mlx.nn.Sigmoid"><span
class="pre"><code class="sourceCode python">Sigmoid</code></span></a>() | Applies the sigmoid function, element-wise. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SiLU.html#mlx.nn.SiLU"
class="reference internal" title="mlx.nn.SiLU"><span class="pre"><code
class="sourceCode python">SiLU</code></span></a>() | Applies the Sigmoid Linear Unit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.SinusoidalPositionalEncoding.html#mlx.nn.SinusoidalPositionalEncoding"
class="reference internal"
title="mlx.nn.SinusoidalPositionalEncoding"><span class="pre"><code
class="sourceCode python">SinusoidalPositionalEncoding</code></span></a>(dims\[, ...\]) | Implements sinusoidal positional encoding. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softmin.html#mlx.nn.Softmin"
class="reference internal" title="mlx.nn.Softmin"><span
class="pre"><code class="sourceCode python">Softmin</code></span></a>() | Applies the Softmin function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softshrink.html#mlx.nn.Softshrink"
class="reference internal" title="mlx.nn.Softshrink"><span
class="pre"><code class="sourceCode python">Softshrink</code></span></a>(\[lambd\]) | Applies the Softshrink function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softsign.html#mlx.nn.Softsign"
class="reference internal" title="mlx.nn.Softsign"><span
class="pre"><code class="sourceCode python">Softsign</code></span></a>() | Applies the Softsign function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softmax.html#mlx.nn.Softmax"
class="reference internal" title="mlx.nn.Softmax"><span
class="pre"><code class="sourceCode python">Softmax</code></span></a>() | Applies the Softmax function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Softplus.html#mlx.nn.Softplus"
class="reference internal" title="mlx.nn.Softplus"><span
class="pre"><code class="sourceCode python">Softplus</code></span></a>() | Applies the Softplus function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Step.html#mlx.nn.Step"
class="reference internal" title="mlx.nn.Step"><span class="pre"><code
class="sourceCode python">Step</code></span></a>(\[threshold\]) | Applies the Step Activation Function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Tanh.html#mlx.nn.Tanh"
class="reference internal" title="mlx.nn.Tanh"><span class="pre"><code
class="sourceCode python">Tanh</code></span></a>() | Applies the hyperbolic tangent function. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Transformer.html#mlx.nn.Transformer"
class="reference internal" title="mlx.nn.Transformer"><span
class="pre"><code
class="sourceCode python">Transformer</code></span></a>(dims, num_heads, ...) | Implements a standard Transformer model. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Upsample.html#mlx.nn.Upsample"
class="reference internal" title="mlx.nn.Upsample"><span
class="pre"><code class="sourceCode python">Upsample</code></span></a>(scale_factor\[, mode, align_corners\]) | Upsample the input signal spatially. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update_modules.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Module.update_modules

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ALiBi.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.ALiBi

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
