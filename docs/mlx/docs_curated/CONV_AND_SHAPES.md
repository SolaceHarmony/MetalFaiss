# Convolutions and Shapes

Convolution layers are sensitive to input layout and shape math. This guide nails the basics to avoid porting mistakes.

## Conventions

- Inputs: `(N, C, H, W)` by default (batch, channels, height, width).
- Groups: `groups > 1` splits channels; depthwise conv is `groups = C` with `out_channels = C`.

## Output Shape

For 2D conv with `kernel_size = (kH, kW)`, `stride = (sH, sW)`, `padding = (pH, pW)`, `dilation = (dH, dW)`,

```text
H_out = floor((H + 2*pH - dH*(kH-1) - 1)/sH + 1)
W_out = floor((W + 2*pW - dW*(kW-1) - 1)/sW + 1)
```

## Examples

```python
import mlx.core as mx
import mlx.nn as nn

x = mx.random.normal((8, 3, 32, 32))
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
y = conv(x)  # (8, 16, 32, 32)

conv_s2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
y2 = conv_s2(y)  # (8, 32, 16, 16)
```

Groups and depthwise:

```python
dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
z = dw(y2)  # (8, 32, 16, 16)
```

Transpose conv upsamples spatially:

```python
deconv = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
u = deconv(y2)  # (8, 16, 32, 32)
```

## Pooling and Flatten

```python
pool = nn.AvgPool2d(2)
p = pool(y)                 # halves H,W
flat = nn.Flatten()
f = flat(p)                 # (8, C*H*W)
```

## Sanity Checks

Always print shapes during bring‑up:

```python
print(x.shape, y.shape, y2.shape)
```

