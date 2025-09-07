# metalfaiss.compat — Wrapper Layer Design

Purpose
- Provide a FAISS‑like Python API on top of MetalFaiss without constraining
  the internal MLX/Metal design.
- Make status obvious: each class/method has a docstring with Implemented /
  Partial / Stub labels and pointers to PLAN.md.

Guidelines
- Wrappers accept Python lists but always return MLX arrays (pure MLX contract).
- No Python math on arrays; use MLX ops (mx.square/divide/multiply/where/...).
- No host pulls (`.item()/.tolist()/.numpy()`) in hot paths.
- Keep compiled MLX functions optional (wrappers should not rebuild compiled
  lambdas in tight loops; cache at module level if used).

Structure (current → future)
- Today: `metalfaiss.faiss_compat` implements the facade; this package re‑exports
  it to allow gradual split:
  - adapters/flat.py, ivf.py, pq.py, hnsw.py, transforms.py
  - adapters/shards.py (stubs), adapters/sq.py (stub/shim)
  - factory.py (index_factory + mapping table)
  - registry.py (feature → status matrix; can generate docs/compat table)

Testing
- Add lightweight tests that instantiate each wrapper, call train/add/search on
  small toy data, and verify shapes + monotonicity for distances/top‑k.
- For stubs, assert NotImplementedError with a helpful message (points to PLAN).

Docs
- Keep docstrings rich (backing class, API deltas, status). The registry can be
  used to generate docs/compat/FAISS-API-Checklist.md.

See also
- PLAN.md — roadmap and upstream reference
- docs/compat/FAISS-API-Checklist.md — status matrix
