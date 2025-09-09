Streams and Banded Execution in MLX

When to Use

- For very large problems, you can split work into independent “bands” and dispatch them on multiple streams to overlap latency and host overhead. This approach helps only when math per band is large enough and memory headroom exists.

MLX Streams API

```python
import mlx.core as mx
dev = mx.default_device()
streams = [mx.new_stream(dev) for _ in range(S)]
outs = [None]*len(bands)
for idx,(s,e) in enumerate(bands):
    st = streams[idx % S]
    with mx.stream(st):
        Vb = V[:, s:e]
        Bb = gemm_av(A, Vb)   # custom kernel
        Zb = gemm_at_b(A, Bb) # custom kernel
        outs[idx] = Zb
mx.synchronize()
Z = mx.concatenate(outs, axis=1)
```

Guidelines

- Keep band shapes stable (compile caches and kernel caches benefit).
- Start with serial bands and measure; add streams only if larger tiles and profiles suggest overlap.
- Use a modest number of streams (2–8); evaluate peak memory usage.
- Prefer two kernels (A@V band; Aᵀ@B band) over monolithic kernels for SVD.

Results Snapshot (example)

- Small/medium shapes often show no gain with streams; serial banding can help at small k, but monolithic tiled kernels are usually competitive.
- For much larger shapes, streams may help; always benchmark with warmup and medians.

References

- See also: Tiled-LinearAlgebra.md and Benchmarks-and-Pruning.md

