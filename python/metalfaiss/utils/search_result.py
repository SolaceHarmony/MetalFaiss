"""
search_result.py - Search result classes for MetalFaiss
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import mlx.core as mx

@dataclass(init=False)
class SearchResult:
    """Result of k-nearest neighbor search."""

    distances: mx.array
    indices: mx.array

    def __init__(self, distances, indices: Optional[mx.array] = None, labels: Optional[Iterable[Iterable[int]]] = None):
        if indices is None and labels is None:
            raise ValueError("Either indices or labels must be provided")
        if indices is None:
            indices = mx.array(labels, dtype=mx.int64)
        self.distances = distances
        self.indices = indices
        self.__post_init__()

    def __post_init__(self):
        if self.distances.shape != self.indices.shape:
            raise ValueError("Distances and indices must have same shape")

    def __len__(self) -> int:
        return len(self.distances)

    @property
    def nq(self) -> int:
        """Number of query vectors."""
        return int(self.distances.shape[0])

    @property
    def k(self) -> int:
        """Number of neighbors per query."""
        return int(self.distances.shape[-1])

    def __getitem__(self, idx) -> 'SearchResult':
        return SearchResult(
            distances=self.distances[idx],
            indices=self.indices[idx]
        )

    @property
    def labels(self) -> mx.array:
        return self.indices

    def __iter__(self):
        yield self.distances
        yield self.indices

@dataclass(init=False)
class SearchRangeResult:
    """Result of range search."""

    distances: List[mx.array]
    indices: List[mx.array]
    lims: mx.array

    def __init__(self, *, distances: Sequence[Sequence[float]] | Sequence[mx.array], lims: Sequence[int], indices: Optional[Sequence[Sequence[int]] | Sequence[mx.array]] = None, labels: Optional[Sequence[Sequence[int]]] = None):
        if indices is None and labels is None:
            raise ValueError("Either indices or labels must be provided")
        self.distances = [dist if isinstance(dist, mx.array) else mx.array(dist, dtype=mx.float32) for dist in distances]
        if indices is None:
            indices = labels if labels is not None else []
        self.indices = [idx if isinstance(idx, mx.array) else mx.array(idx, dtype=mx.int64) for idx in indices]
        self.lims = mx.array(lims, dtype=mx.int32)
        self.__post_init__()

    def __post_init__(self):
        if len(self.distances) != len(self.indices):
            raise ValueError("Must have same number of distance and index arrays")
        if len(self.lims) != len(self.distances) + 1:
            raise ValueError("Lims array must have length n_queries + 1")

    def __len__(self) -> int:
        return len(self.distances)

    @property
    def nq(self) -> int:
        return len(self.distances)

    def __getitem__(self, idx) -> tuple:
        dist = self.distances[idx]
        ind = self.indices[idx]
        if isinstance(dist, mx.array) and int(dist.shape[0]) == 0:
            return [], []
        return dist, ind

    @property
    def labels(self) -> List[mx.array]:
        return self.indices

    def get_total_size(self) -> int:
        return int(self.lims[-1])

    def merge(self, other: 'SearchRangeResult') -> 'SearchRangeResult':
        if self.nq != other.nq:
            raise ValueError("Range results must have the same number of queries to merge")

        merged_distances: List[mx.array] = []
        merged_indices: List[mx.array] = []
        merged_lims = [0]

        for d1, i1, d2, i2 in zip(self.distances, self.indices, other.distances, other.indices):
            dcat = mx.concatenate([d1, d2])
            icat = mx.concatenate([i1, i2])
            if int(dcat.shape[0]) > 0:
                order = mx.argsort(dcat)
                dsorted = mx.take(dcat, order, axis=0)
                isorted = mx.take(icat, order, axis=0)
            else:
                dsorted = dcat
                isorted = icat
            merged_distances.append(dsorted)
            merged_indices.append(isorted)
            merged_lims.append(merged_lims[-1] + int(dsorted.shape[0]))

        return SearchRangeResult(
            distances=merged_distances,
            indices=merged_indices,
            lims=mx.array(merged_lims, dtype=mx.int32)
        )

    def to_arrays(self) -> tuple[mx.array, mx.array, mx.array]:
        distances_flat = mx.concatenate(self.distances) if self.distances else mx.array([], dtype=mx.float32)
        indices_flat = mx.concatenate(self.indices) if self.indices else mx.array([], dtype=mx.int64)
        return self.lims, distances_flat, indices_flat
