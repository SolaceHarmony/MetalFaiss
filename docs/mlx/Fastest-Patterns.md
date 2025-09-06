Fastest Implementation Patterns (Practical, Portable Playbook)

This is a plain‑speech checklist of what consistently measures fastest for GPU‑heavy MLX + Metal (and CUDA) projects. The ideas are broadly useful beyond this repo.

Principles
- Move less, fuse more: read once, compute more, write once. Fusing distance + selection or tile load + FMA beats multiple passes through memory.
- Batch to amortize: launch overhead matters. Batch queries or tiles where results are independent.
- Choose the smallest correct scope: use threadgroup scope for data that crosses warps; use warp‑local ops when sharing stays within the warp.
- Static tuning over autotune: select per‑device parameters offline; ship a static config (env overrides for experts).
- Overlap with streams: run independent work on explicit streams; synchronize only at natural boundaries.
- Profile, then lock: measure on target devices, pick winners, and persist in config.

Kernel‑Level Patterns
- Fused scan + top‑k: when searching lists (e.g., IVF), compute distances and maintain a local top‑k inside the kernel. Avoid materializing full distance arrays.
  - Maintain a small per‑thread local heap (k ≤ 32), then reduce across the threadgroup to emit the final top‑k.
  - For very long lists, do a two‑pass device merge: multiple threadgroups produce partial (k) lists, then a device merge kernel selects the final (k).
- Warp‑level reductions: when doing dot products or per‑column accumulations, assign one warp per result and stride lanes across the reduction dimension. Reduce within the warp (simdgroup) or via threadgroup scratch.
  - Heuristic: switch to the warp‑reduction kernel when the reduction dimension is large (e.g., m ≥ 512 for QR projection c = Qᵀv).
- Batched queries: if many queries use the same candidate set (e.g., same probed lists), launch one threadgroup per query with a shared X; this can be an order of magnitude faster than sequential single‑query launches.
- Tiling + fma: for matrix‑like ops, use 2D tiling, stage tiles in threadgroup memory, and accumulate with fma. Keep tile sizes device‑aware and multiples of the execution width.
- Avoid slow ops in hot loops: no integer division/modulus by runtime values; remove dynamic stack arrays; use half I/O + float accumulation if error tolerance allows.
- Barrier discipline: barrier only when data crosses warps and only with the required memory scope. Don’t over‑synchronize.

System‑Level Patterns
- Static params, no autotune: pick tile sizes, QR dot mode, band sizes, and stream counts per device and store in a config file.
  - This repo: `faissmlx/config/hardware_params.json` loaded by `faissmlx/tuning.py`.
  - Precedence: env override → static config → heuristic default.
- Streams and callbacks: place independent work on explicit streams; rely on MLX’s cross‑stream dependencies instead of global fences. Use a background waiter to run host callbacks when a stream or arrays complete.
  - Helpers: `python/metalfaiss/utils/streams.py`.
- Chunk if you must, merge on device: where problems exceed useful single‑kernel sizes, chunk the input and do a device‑side merge rather than host‑side.

Practical Defaults & Knobs
- Tiles (GEMM): loaded from config; defaults on Apple M3: AV(32×8), AT_B(8×32); others: 16×16. Env overrides:
  - `METALFAISS_GEMM_TILE_AV=TMxT`, `METALFAISS_GEMM_TILE_ATB=TNxTK`
- QR projection mode: auto‑select simple vs warp‑reduction; env override:
  - `METALFAISS_QR_DOT=simple|simd`
- IVF fused scan + select: enable fused kernel and control threads per block:
  - `METALFAISS_USE_IVF_TOPK=1`, `METALFAISS_IVF_TPB=<threads>` (safe max auto‑selects)
- SVD Z‑step banding and streams: band small k and use a small number of streams:
  - `METALFAISS_SVD_BAND=<B>`, `METALFAISS_SVD_STREAMS=<S>`

Case Studies (Measured Here)
- QR projection (c = Qᵀv): warp‑reduction kernel (~32‑lane) beats per‑thread loops for large m; correctness matches MLX dot.
- IVF search:
  - Fused concat (one kernel across probed lists) matches or slightly beats MLX arg‑sort for d=64, N≈32k, nprobe∈{1,8}.
  - Per‑list kernels with host merge are slower (launch + merge overhead).
  - Chunk + device merge approaches fused concat and scales to longer lists.
  - Batched (same X) is 1–2 orders faster for Q=16 in our tests; batch when candidate sets are shared.
- SVD Z‑step: tiled two‑pass (A@V then Aᵀ@B) beats MLX for small k at moderate sizes; banding wins for small k; streams offer modest gains when memory permits.

Checklist
- [ ] Data moves minimized and fused where safe
- [ ] Warp‑level reductions for large reductions
- [ ] Batched launches where candidates are shared
- [ ] Tile sizes set per device from static config
- [ ] No integer division/modulus in hot loops
- [ ] Barrier scope minimal and correct
- [ ] Streams explicit; only boundary‑level sync
- [ ] Benchmarked, then persisted in config; env overrides documented

