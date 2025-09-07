# MetalFaiss — FAISS API Compatibility Plan

Purpose
- Track the concrete steps to provide FAISS API surface compatibility via light wrappers and shims (no redesign of our MLX/Metal internals).
- Keep a quick pointer to the upstream FAISS repo used for reference.

Reference
- Upstream FAISS checkout (local): `/Volumes/stuff/Projects/faiss`
  - Top-level CMake targets and subdirs: `faiss/`, `faiss/gpu`, `c_api/`, `faiss/python`, `benchs/`, `demos/`, `tests/`

Where we are (quick status)
- Implemented (MLX/Metal):
  - Flat (L2/IP), IVFFlat (L2), IVFPQ (ADC), PQ, HNSW, LSH, Binary*(Flat/IVF/HNSW)
  - IDMap/IDMap2, PreTransform, RefineFlat
  - Transforms: PCA/OPQ/Normalization; clustering (k‑means); distances/norms
- Partially implemented / wrappers needed:
  - IVFOPQ wrapper (compose OPQ→IVFPQ under one class)
  - ScalarQuantizer (real quantization paths) or a documented shim
  - Range search API (Flat/IVF*/PQ) with pure‑MLX SearchRangeResult
  - Remove/Reconstruct APIs across Flat/IVF*/PQ
  - IndexShards/IndexReplicas wrappers (fan‑out adds; device‑merge top‑k)
  - Index factory string coverage (OPQm_d,IVFn,PQk, SQ8, HNSW, etc.)
  - IO parity: read_index/write_index without NumPy; purge remaining host conversions

Design stance
- We do not re‑implement FAISS internals in C++; we expose compatible class/function names backed by our MLX/Metal code. Thin wrappers are acceptable.

Deliverables (checklist)

1) Compatibility Facade (faiss_compat.py)
- [ ] Add `metalfaiss/faiss_compat.py` exporting FAISS‑like names:
  - Classes: `IndexFlatL2`, `IndexFlatIP`, `IndexIVFFlat`, `IndexIVFPQ`, `IndexPQ`, `IndexHNSWFlat`, `IndexIDMap`, `IndexIDMap2`, `IndexPreTransform`, `IndexRefineFlat`, `IndexShards`, `IndexReplicas`, `IndexIVFScalarQuantizer` (shim), `IndexIVFOPQ` (wrapper)
  - Functions: `index_factory(d, descr)`, `normalize_L2(x)`, `read_index(path)`, `write_index(path)`
  - Param shims: `setNumProbes` (nprobe), `set_efSearch`, `set_efConstruction`

2) IVFOPQ Wrapper
- [ ] New class composing OPQ → IVFPQ
  - train(xs): fit OPQ; transform; train IVFPQ
  - add/search: transform then delegate to IVFPQ
- [ ] Extend factory to parse `OPQm_d,IVF{n},PQ{k}` strings

3) ScalarQuantizer (SQ)
- [ ] Implement per‑dim codebooks (scale/zero‑point) and search
  - OR: provide clear shim → Flat with warning + TODO label
- [ ] Update `index/scalar_quantizer_index.py` (now returns MLX arrays)

4) Range Search (pure MLX)
- [ ] Add `range_search(x, radius)` to Flat; return MLX SearchRangeResult (dists/labels/lims)
- [ ] Implement IVFFlat/IVFPQ range search; keep device‑side selection when possible
- [ ] Purge NumPy from `utils/range_search.py` & `search_range_result.py`

5) Removal/Reconstruction
- [ ] `remove_ids(ids)` and `reconstruct(i)/reconstruct_n(i, n)` for Flat/IVFFlat/IVFPQ
- [ ] Maintain inverted lists consistency on remove

6) Shards/Replicas
- [ ] `IndexShards` wrapper (split add; merge top‑k on device)
- [ ] `IndexReplicas` wrapper (mirror add; merge top‑k)

7) Factory Coverage
- [ ] Parse and map: `Flat`, `HNSW{M},Flat`, `IVF{n},Flat`, `IVF{n},PQ{k}`, `OPQ{m}_{d},IVF{n},PQ{k}`, `IVF{n},SQ8`
- [ ] Validate unsupported patterns → clear errors & fallbacks

8) IO Parity (MLX‑only)
- [ ] `read_index`/`write_index` for all above; no `.numpy()`; document format
- [ ] Replace remaining NumPy in IO helpers (`binary_io.py`, etc.)

9) Tests & Benches
- [ ] Unit tests for wrappers (surface parity & basic functionality)
- [ ] Range search tests (dists monotonicity; lims correctness)
- [ ] Small parity benchmarks (Flat/IVF*/IVFPQ) — re‑use `docs/benchmarks` harness; auto‑archive enabled

10) Docs
- [ ] `docs/compat/FAISS-API-Checklist.md` — matrix of FAISS features → status/notes
- [ ] Quickstart for users coming from FAISS (how to import the compat facade)

Notes / context reload
- Upstream FAISS root: `/Volumes/stuff/Projects/faiss` — run `sed -n '1,200p' CMakeLists.txt` to rediscover the build graph if context is lost.
- Our MLX compile integration: `docs/mlx/Compile-Guide.md`; curated copy at `agent_knowledgebase/mlx/mlx.core.compile.md` (+ guides/MetalFaiss-Compile-Guide.md)
- Bench harness writes CSV/PNG to `docs/benchmarks/` and auto‑archives prior runs to `docs/benchmarks/archive/<timestamp>`.

Definition of Done (compat phase)
- The compat facade supports standard FAISS recipes:
  - `IndexFlatL2/IP`, `IndexIVFFlat` (train/add/search), `IndexIVFPQ` (ADC), `IndexHNSWFlat` (ef setters)
  - `IndexIVFOPQ` wrapper
  - `IndexIDMap/IDMap2`, `IndexPreTransform`, `IndexRefineFlat`, `IndexShards/Replicas`
  - `index_factory` strings for common pipelines
  - `normalize_L2`, `read_index`/`write_index`
- All return types are MLX arrays; no host conversions on hot paths.
