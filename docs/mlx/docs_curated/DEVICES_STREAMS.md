# Devices and Streams

MLX runs on Metal by default when available and exposes devices and streams for explicit control over placement and concurrency.

Key concepts
- Asynchronous ops: streams queue work without blocking Python; evaluation or explicit sync waits for results.
- Device association: each stream is tied to a specific device; ops dispatched to a stream execute on that device.
- Default stream: by default, ops run on the default stream of `mx.default_device()`.
- Dependencies: MLX tracks data dependencies. If an op needs results produced on another stream, MLX orders execution so data is ready.

## Devices

```python
import mlx.core as mx

print(mx.default_device())  # e.g., Device(gpu:0)

with mx.default_device(mx.cpu):
    x = mx.ones((1024, 1024))

# CPU route for a single op without changing global default
mx.set_default_device(mx.gpu)  # if available
y64 = mx.sum(x.astype(mx.float64), stream=mx.cpu)
```

Utilities:

- `mx.set_default_device(device)`: set globally
- `mx.get_active_memory()` / `mx.get_peak_memory()`
- `mx.reset_peak_memory()`
- Float64 is CPU‑only; scope double‑precision work via `with mx.default_device(mx.cpu)` or per‑op `stream=mx.cpu`.
- Many ops don’t take `device=`; control placement via default/scoped device or stream.

## Streams

Streams allow overlapping compute and host logic.

```python
s = mx.new_stream()
with mx.stream(s):
    y = mx.random.normal((4096, 4096)) @ mx.random.normal((4096, 4096))
    # host continues

mx.synchronize()  # wait for outstanding device work

# Per‑op stream example
z = mx.random.normal((1024, 1024), stream=s)
```

Core APIs
- `mx.default_device()` / `mx.set_default_device(device)`
- `mx.default_stream(device)` / `mx.set_default_stream(stream)`
- `mx.new_stream(device=None)` → create an independent stream (defaults to current device)
- `mx.stream(stream)` → context manager to set default device/stream
- `mx.synchronize(stream=None)` → block until all work on stream (or all work) completes

Most users start without streams; add them as workloads grow.

Pitfalls and tips:
- Lazy execution: work enqueues until evaluation; synchronize only when host needs results.
- Avoid excessive stream switching; batch related ops within a single stream block.
- Route unsupported ops to CPU streams (see Backend Support), keeping the rest on GPU.

## Example: CPU and GPU in parallel

```python
import mlx.core as mx

# Two inputs
a = mx.random.uniform(shape=(2048, 2048))
b = mx.random.uniform(shape=(2048, 2048))

# Create streams bound to CPU and GPU (if available)
cpu_stream = mx.new_stream(mx.cpu)
gpu_stream = mx.new_stream(mx.gpu)

# Dispatch independent ops; nothing runs yet (lazy)
c = mx.add(a, b, stream=cpu_stream)
d = mx.matmul(a, b, stream=gpu_stream)

# Wait for each stream if you need the results now
mx.synchronize(cpu_stream)
mx.synchronize(gpu_stream)

# Materialize arrays (or continue passing them through MLX ops)
mx.eval(c, d)
```

Notes
- Use `.item()` only on scalars (0‑D arrays). For matrices/vectors, `mx.eval` materializes results while keeping them in MLX.
- Because MLX manages dependencies, if `d` depends on `c`, you can skip manual stream syncs and just `mx.eval(d)`.

## Metal Capture (Advanced)

For profiling or recording command buffers:

- `mx.metal.start_capture()` / `mx.metal.stop_capture()`
- `mx.metal.device_info()` for capabilities

Use only when diagnosing performance; it’s not needed for training loops.

## Troubleshooting

- If results differ across runs, print `mx.default_device()` and check where eval happens.
- For performance anomalies, try `mx.clear_cache()` sparingly, and prefer compiled, pure graphs when available.

## For NumPy Users

- NumPy has no device concept; MLX arrays may live on GPU/CPU. Use `with mx.default_device(...)` to scope placement.
- Host scalars: call `.item()` to bring a 0‑D array to Python (useful for logging).
- Streams are MLX/Metal‑specific for concurrency; keep it simple until profiling shows need.
