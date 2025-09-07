"""
metalfaiss.compat — FAISS‑like wrapper layer (API surface)

This package provides a familiar FAISS Python API that delegates to the
MetalFaiss MLX/Metal implementations. It exists to ease migration without
constraining internal designs.

Patterns
- Thin class wrappers with a private `_impl` that holds our native index.
- Comprehensive docstrings per class/method indicating status:
  Implemented / Partial / Stub (NotImplementedError) with a pointer to PLAN.md.
- Pure MLX contract: methods accept Python lists but always return MLX arrays.
- No Python math or host pulls; constants are MLX scalars.

Modules
- The initial implementation is in `metalfaiss.faiss_compat`.
  We re‑export symbols here to allow future split by adapter modules
  (flat.py, ivf.py, pq.py, hnsw.py, shards.py, sq.py, factory.py).

See also
- PLAN.md for the compatibility roadmap
- docs/compat/FAISS-API-Checklist.md for a status matrix
"""

from ..faiss_compat import *  # re-export current facade

__all__ = [name for name in dir() if not name.startswith("_")]

