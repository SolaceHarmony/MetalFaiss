"""
faiss_compat.py — FAISS API surface wrappers (stubs + adapters) for MetalFaiss

Purpose
- Provide a familiar FAISS-like Python API while delegating to our pure‑MLX/Metal
  implementations. This file documents what is implemented vs. stubbed so teams
  can see progress at a glance and code against stable names.

Conventions
- All methods return MLX arrays (no NumPy); inputs can be Python lists which are
  converted to MLX inside the underlying index.
- If a feature is not implemented yet, methods raise NotImplementedError with a
  clear TODO. Where safe, shims fall back to simpler behavior and log a note in
  the docstring.

Key classes (status)
- IndexFlatL2/IP          — Implemented (wraps FlatIndex)
- IndexIVFFlat            — Implemented (wraps IVFFlatIndex)
- IndexIVFPQ              — Implemented (wraps IVFPQIndex)
- IndexPQ                 — Implemented (wraps ProductQuantizerIndex)
- IndexHNSWFlat           — Implemented (wraps HNSWIndex), exposes ef params
- IndexIDMap/IDMap2       — Implemented (wrap ID map wrappers)
- IndexPreTransform       — Implemented (wraps PreTransformIndex)
- IndexRefineFlat         — Implemented (wraps RefineFlatIndex)
- IndexIVFOPQ             — Stub (planned: OPQ→IVFPQ composition)
- IndexIVFScalarQuantizer — Stub/shim (planned: real SQ codebooks)
- IndexShards/IndexReplicas — Stubs (planned: split/mirror and device‑merge top‑k)

Functions (status)
- index_factory           — Partial mapping; extend incrementally
- normalize_L2            — Implemented (pure MLX)
- write_index/read_index  — Partial (delegates to index_io; will be MLX‑only)

Note: This file is for API parity only — it does not change core performance
paths. Wrappers call into modules under metalfaiss.index / vector_transform.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import mlx.core as mx

from .types.metric_type import MetricType
from .index.flat_index import FlatIndex
from .index.ivf_flat_index import IVFFlatIndex
from .index.ivf_pq_index import IVFPQIndex
from .index.product_quantizer_index import ProductQuantizerIndex
from .index.hnsw_index import HNSWIndex
from .index.id_map import IDMap
from .index.id_map2 import IDMap2
from .index.pre_transform_index import PreTransformIndex
from .index.refine_flat_index import RefineFlatIndex
from .utils.search_result import SearchResult


# ------------ Simple adapters (implemented) ------------

class IndexFlatL2:
    """FAISS‑like IndexFlatL2 wrapper (Implemented).

    Backed by: metalfaiss.index.flat_index.FlatIndex(metric=L2)
    Implemented: train (no‑op), add, search, ntotal property.
    """

    def __init__(self, d: int):
        self._impl = FlatIndex(d, metric_type=MetricType.L2)

    @property
    def d(self) -> int:
        return self._impl.d

    @property
    def ntotal(self) -> int:
        return self._impl.ntotal

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]]) -> None:
        self._impl.add(xs)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        vals, idx = self._impl.search(xs, k)
        return SearchResult(distances=vals, indices=idx)


class IndexFlatIP:
    """FAISS‑like IndexFlatIP wrapper (Implemented).

    Backed by: metalfaiss.index.flat_index.FlatIndex(metric=INNER_PRODUCT)
    """

    def __init__(self, d: int):
        self._impl = FlatIndex(d, metric_type=MetricType.INNER_PRODUCT)

    @property
    def d(self) -> int:
        return self._impl.d

    @property
    def ntotal(self) -> int:
        return self._impl.ntotal

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]]) -> None:
        self._impl.add(xs)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        vals, idx = self._impl.search(xs, k)
        return SearchResult(distances=vals, indices=idx)


class IndexIVFFlat:
    """FAISS‑like IndexIVFFlat (Implemented).

    Backed by: metalfaiss.index.ivf_flat_index. Supports L2 metric, nlist/nprobe.
    API:
      - train(xs), add(xs[, ids]), search(xs, k)
      - properties: nlist, nprobe
    """

    def __init__(self, quantizer: IndexFlatL2, d: int, nlist: int, metric: str = "L2"):
        # quantizer is accepted for parity; we construct our own FlatIndex internally.
        self._impl = IVFFlatIndex(FlatIndex(d, metric_type=MetricType.L2), d, nlist)

    @property
    def nlist(self) -> int:
        return self._impl.nlist

    @property
    def nprobe(self) -> int:
        return self._impl.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._impl.nprobe = value

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        self._impl.add(xs, ids)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexIVFPQ:
    """FAISS‑like IndexIVFPQ (Implemented).

    Backed by: metalfaiss.index.ivf_pq_index. L2 metric only.
    API: train, add, search; nlist/nprobe properties.
    """

    def __init__(self, d: int, nlist: int, M: int, nbits: int = 8):
        self._impl = IVFPQIndex(d, nlist, M, nbits, metric_type=MetricType.L2)

    @property
    def nlist(self) -> int:
        return self._impl.nlist

    @property
    def nprobe(self) -> int:
        return self._impl.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._impl.nprobe = value

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        self._impl.add(xs, ids)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexPQ:
    """FAISS‑like IndexPQ (Implemented; exhaustive PQ search).

    Backed by: metalfaiss.index.product_quantizer_index.ProductQuantizerIndex
    """

    def __init__(self, d: int, M: int, nbits: int = 8):
        self._impl = ProductQuantizerIndex(d, M, nbits)

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]]) -> None:
        self._impl.add(xs)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexHNSWFlat:
    """FAISS‑like IndexHNSWFlat (Implemented).

    Backed by: metalfaiss.index.hnsw_index.HNSWIndex
    Exposes: efConstruction, efSearch setters (mapped to underlying fields).
    """

    def __init__(self, d: int, M: int = 32):
        self._impl = HNSWIndex(d, M, metric_type=MetricType.L2)

    @property
    def efSearch(self) -> int:
        return int(self._impl.hnsw.efSearch)

    @efSearch.setter
    def efSearch(self, v: int) -> None:
        self._impl.hnsw.efSearch = int(v)

    @property
    def efConstruction(self) -> int:
        return int(self._impl.hnsw.efConstruction)

    @efConstruction.setter
    def efConstruction(self, v: int) -> None:
        self._impl.hnsw.efConstruction = int(v)

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]]) -> None:
        self._impl.add(xs)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexIDMap:
    """FAISS‑like IndexIDMap (Implemented). Wraps base index with explicit ids."""

    def __init__(self, base):
        self._impl = IDMap(base)

    def add_with_ids(self, xs: List[List[float]], ids: List[int]) -> None:
        self._impl.add_with_ids(xs, ids)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexIDMap2:
    """FAISS‑like IndexIDMap2 (Implemented)."""

    def __init__(self, base):
        self._impl = IDMap2(base)

    def add_with_ids(self, xs: List[List[float]], ids: List[int]) -> None:
        self._impl.add_with_ids(xs, ids)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexPreTransform:
    """FAISS‑like IndexPreTransform (Implemented)."""

    def __init__(self, vt, base):
        self._impl = PreTransformIndex(vt, base)

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]]) -> None:
        self._impl.add(xs)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


class IndexRefineFlat:
    """FAISS‑like IndexRefineFlat (Implemented)."""

    def __init__(self, base, refine):
        self._impl = RefineFlatIndex(base, refine)

    def train(self, xs: List[List[float]]) -> None:
        self._impl.train(xs)

    def add(self, xs: List[List[float]]) -> None:
        self._impl.add(xs)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self._impl.search(xs, k)


# ------------ Planned wrappers / stubs ------------

class IndexIVFOPQ:
    """FAISS‑like IndexIVFOPQ (Stub).

    Intended behavior:
      - Compose OPQ transform (m,d) with an IVFPQ index under the hood.
      - train(xs): fit OPQ on xs; transform xs; train IVFPQ on residuals.
      - add/search: transform inputs, delegate to IVFPQ.

    Status: stub — raises NotImplementedError. See PLAN.md for milestones.
    """

    def __init__(self, d: int, nlist: int, M: int, nbits: int = 8, m: int = 16, d_out: Optional[int] = None):
        raise NotImplementedError("IndexIVFOPQ is not implemented yet. See PLAN.md → IVFOPQ Wrapper.")


class IndexIVFScalarQuantizer:
    """FAISS‑like IndexIVFScalarQuantizer (Stub/Shim).

    Intended behavior:
      - Scalar quantization per dimension with codebooks; IVF lists & distance on codes.

    Current shim: use IVFFlat for compute and store raw vectors or raise a clear
    error if strict SQ is required.
    """

    def __init__(self, d: int, nlist: int, qtype: str = "QT_8bit"):
        raise NotImplementedError("IVFScalarQuantizer is a stub. Use IVFFlat or see PLAN.md → ScalarQuantizer.")


class IndexShards:
    """FAISS‑like IndexShards (Stub).

    Intended behavior:
      - Split add() across child shards; merge top‑k on search (device‑side merge preferred).
    """

    def __init__(self, d: int, threaded: bool = False):
        self.d = d
        self.threaded = threaded
        self.children: List[object] = []

    def add_shard(self, index) -> None:
        self.children.append(index)

    def add(self, xs: List[List[float]]) -> None:
        raise NotImplementedError("IndexShards.add not implemented. See PLAN.md → Shards/Replicas.")

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        raise NotImplementedError("IndexShards.search not implemented. See PLAN.md → Shards/Replicas.")


class IndexReplicas:
    """FAISS‑like IndexReplicas (Stub).

    Intended behavior:
      - Mirror add() to all replicas; merge top‑k on search.
    """

    def __init__(self, d: int):
        self.d = d
        self.children: List[object] = []

    def add_replica(self, index) -> None:
        self.children.append(index)

    def add(self, xs: List[List[float]]) -> None:
        raise NotImplementedError("IndexReplicas.add not implemented. See PLAN.md → Shards/Replicas.")

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        raise NotImplementedError("IndexReplicas.search not implemented. See PLAN.md → Shards/Replicas.")


# ------------ Utility functions ------------

def normalize_L2(x: mx.array) -> mx.array:
    """In‑place L2 normalization (pure MLX).

    x: (n,d) → returns (n,d), rows normalized. Avoids div by zero via eps guard.
    """
    norms = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True))
    norms = mx.maximum(norms, mx.array(1e-20, dtype=x.dtype))
    return mx.divide(x, norms)


def index_factory(d: int, description: str):
    """Partial FAISS index_factory implementation.

    Supports common strings now; extend in small steps:
      - "Flat" → IndexFlatL2
      - "FlatIP" → IndexFlatIP
      - "IVF{n},Flat" → IndexIVFFlat(nlist=n)
      - "IVF{n},PQ{M}" → IndexIVFPQ(nlist=n, M=M)
      - "HNSW{M},Flat" → IndexHNSWFlat(M)
    TODO: "OPQ{m}_{d},IVF{n},PQ{M}", "IVF{n},SQ8".
    """
    s = description.replace(" ", "")
    if s == "Flat":
        return IndexFlatL2(d)
    if s == "FlatIP":
        return IndexFlatIP(d)
    if s.startswith("HNSW") and s.endswith(",Flat"):
        M = int(s.split(",")[0][4:])
        return IndexHNSWFlat(d, M=M)
    if s.startswith("IVF") and s.endswith(",Flat"):
        nlist = int(s.split(",")[0][3:])
        return IndexIVFFlat(IndexFlatL2(d), d, nlist)
    if s.startswith("IVF") and ",PQ" in s:
        head, tail = s.split(",")
        nlist = int(head[3:])
        M = int(tail[2:])
        return IndexIVFPQ(d, nlist, M)
    raise ValueError(f"Unsupported factory string: {description}")


def write_index(index, path: str) -> None:
    """Serialize index to disk (partial).

    Delegates to metalfaiss.index.index_io when available; will be migrated to
    MLX‑only IO (no NumPy). Stub may not cover all index types yet.
    """
    from .index.index_io import write_index as _w
    _w(index, path)


def read_index(path: str):
    """Deserialize index from disk (partial). See note in write_index."""
    from .index.index_io import read_index as _r
    return _r(path)


__all__ = [
    # classes
    "IndexFlatL2", "IndexFlatIP", "IndexIVFFlat", "IndexIVFPQ", "IndexPQ",
    "IndexHNSWFlat", "IndexIDMap", "IndexIDMap2", "IndexPreTransform",
    "IndexRefineFlat", "IndexIVFOPQ", "IndexIVFScalarQuantizer",
    "IndexShards", "IndexReplicas",
    # functions
    "normalize_L2", "index_factory", "write_index", "read_index",
]

