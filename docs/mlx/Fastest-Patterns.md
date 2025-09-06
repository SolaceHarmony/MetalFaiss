Fastest Implementation Patterns (Practical, Portable Playbook)

This is a code‑first guide to the patterns that measured fastest for GPU‑heavy MLX projects (Metal and CUDA). Copy the snippets, tweak the knobs, and profile on your device.

Principles
- Move less, fuse more.
- Batch to amortize.
- Smallest sync scope.
- Static tuning, not autotune.
- Overlap with streams.

Recipe 1 — IVF fused scan + select (single query)
Compute distances and top‑k inside a single kernel across concatenated probed lists.

```python
import mlx.core as mx
from python.metalfaiss.faissmlx.kernels.ivf_kernels import ivf_list_topk_l2

# Inputs
# q: (d,), X: (m,d) concatenated from probed lists, ids: (m,) global ids
vals, idxs = ivf_list_topk_l2(q, X, ids, k=32)  # k<=32, auto tpb; override METALFAISS_IVF_TPB
```

Why it’s fast
- Avoids materializing full distance arrays; selection stays in‑kernel.

Recipe 2 — IVF batched (shared candidates)
If many queries share the same X, launch one threadgroup per query.

```python
from python.metalfaiss.faissmlx.kernels.ivf_kernels import ivf_list_topk_l2_batch

Q = mx.random.normal(shape=(B, d)).astype(mx.float32)
vals, ids = ivf_list_topk_l2_batch(Q, X, ids, k=32)  # returns (B,k)
```

Why it’s fast
- Amortizes launch overhead; measured 10–100× speedups when candidates are shared.

Recipe 3 — IVF chunk + device merge (very long lists)
Split rows, compute per‑chunk top‑k with the fused kernel, and merge on device.

```python
from python.metalfaiss.faissmlx.kernels.ivf_kernels import ivf_list_topk_l2_chunked_device_merge

vals, ids = ivf_list_topk_l2_chunked_device_merge(q, X, ids, k=32, rows_per_chunk=8192)
```

When to use
- Candidate sets exceed useful single‑kernel sizes; on‑device merge outperforms host merge.

Recipe 4 — QR projection with warp‑level reduction
Each warp computes one column of c = Qᵀv. Lanes stride rows, then reduce.

```python
import os
from python.metalfaiss.faissmlx.kernels.qr_kernels import project_coeffs

os.environ["METALFAISS_QR_DOT"] = "simd"  # force warp kernel (auto uses m>=512 heuristic)
c = project_coeffs(Q, v)  # (k,)
```

Why it’s fast
- Reduces per‑thread work and hides latency; correctness matches MLX dot.

Recipe 5 — Tiled GEMM with static tile selection
Use 2D tiles staged in threadgroup memory, accumulate via fma, and set tiles from config.

```python
from python.metalfaiss.faissmlx.kernels import gemm_kernels as gk

print("tiles:", gk.get_gemm_tiles())      # ((TM,T),(TN,TI,TK))
gk.set_gemm_tiles(av="32x8", atb="8x32")  # or via env METALFAISS_GEMM_TILE_*

B = gk.gemm_av(A, V)   # (m,k) = (m,n)@(n,k)
Z = gk.gemm_at_b(A, B) # (n,k) = Aᵀ @ B
```

Why it’s fast
- Coalesced loads to threadgroup tiles + fma in the inner loop maximizes arithmetic intensity.

Recipe 6 — Avoid / and % in hot loops
Map indices without division/modulus by non‑constants. Prefer 2D grids.

```cpp
// Bad (1D mapping): slow / and % by runtime k
uint gid = thread_position_in_grid.x; uint col = gid % k; uint row = gid / k;

// Good (2D): no divide/mod
uint2 g = thread_position_in_grid.xy; if (g.x>=k || g.y>=n) return;
uint idx = g.y * k + g.x;
```

Recipe 7 — Streams overlap + callbacks
Run independent work on explicit streams; fire host callbacks when a stream completes without stalling others.

```python
from python.metalfaiss.utils.streams import on_stream_complete
s_cpu = mx.new_stream(mx.cpu); s_gpu = mx.new_stream(mx.gpu)
with mx.stream(s_cpu): x = preprocess(batch)        # CPU stream
with mx.stream(s_gpu): y = model_forward(x)         # GPU stream
on_stream_complete(s_gpu, lambda: log_ready(y))     # stream‑scoped waiter
```

Knobs & Defaults
- Tiles (GEMM): Apple M3 → AV(32×8), AT_B(8×32); others → 16×16.
  - Env: `METALFAISS_GEMM_TILE_AV=TMxT`, `METALFAISS_GEMM_TILE_ATB=TNxTK`
- QR mode: heuristic or `METALFAISS_QR_DOT=simple|simd`
- IVF fused: `METALFAISS_USE_IVF_TOPK=1`, threads per block `METALFAISS_IVF_TPB`
- SVD banding/streams: `METALFAISS_SVD_BAND`, `METALFAISS_SVD_STREAMS`

Measured outcomes (this repo)
- IVF fused concat matches/slightly beats MLX baseline at d=64, N≈32k, nprobe∈{1,8}.
- IVF batched (shared X) is 1–2 orders faster for Q=16; use when probed lists are shared.
- QR warp‑reduction kernel beats per‑thread loops for large m with correct results.
- SVD Z‑step: tiled two‑pass wins for small k at moderate sizes; banding wins for small k; streams offer modest gains.

Checklist
- [ ] Fused kernels replace multi‑pass memory flows
- [ ] Warp reductions for large reductions
- [ ] Batched launches for shared candidates
- [ ] Tiles from static config; no autotune in prod
- [ ] No div/mod in hot loops
- [ ] Minimal barrier scope
- [ ] Streams explicit; boundary‑only sync
- [ ] Results measured and persisted; env overrides documented
