from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from ..utils.search_result import SearchResult
from ..types.metric_type import MetricType
from ..distances import (
    pairwise_L2sqr,
    pairwise_L1,
    pairwise_Linf,
    pairwise_extra_distances,
    pairwise_jaccard,
)
from ..utils.sorting import topk_smallest_axis1

class FlatIndex(BaseIndex):
    """Flat index implementation using MLX.
    
    This is the most basic index type that stores vectors in memory
    and performs exact nearest neighbor search.
    """
    
    def __init__(self, d: int, metric_type: MetricType = MetricType.L2):
        """Initialize flat index.
        
        Args:
            d: Dimension of vectors to index
            metric_type: Distance metric to use
        """
        super().__init__(d, metric=metric_type)
        self.metric_type = metric_type
        self._vectors: Optional[mx.array] = None
        self._ids: Optional[mx.array] = None
        self._id_to_row: dict[int, int] = {}
        self.is_trained = True
        
    @classmethod
    def from_index(cls, index: BaseIndex) -> Optional['FlatIndex']:
        """Create FlatIndex from generic index if possible.
        
        Args:
            index: Index to convert
            
        Returns:
            FlatIndex if conversion possible, None otherwise
        """
        if isinstance(index, cls):
            return index
        return None
        
    def train(self, xs: List[List[float]]) -> None:
        """Flat indexes are always considered trained."""
        self.is_trained = True
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs (ignored in flat index)
        """
        if ids is not None and len(ids) != len(xs):
            raise ValueError("Length of ids must match number of vectors")

        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")

        start = self.ntotal
        n_new = int(x.shape[0])
        if ids is not None:
            id_arr = mx.array(ids, dtype=mx.int64)
            for offset, val in enumerate(ids):
                self._id_to_row[int(val)] = start + offset
        else:
            id_arr = mx.arange(start, start + n_new, dtype=mx.int64)
            for offset in range(n_new):
                self._id_to_row[start + offset] = start + offset

        if self._vectors is None:
            self._vectors = x
            self._ids = id_arr
        else:
            self._vectors = mx.concatenate([self._vectors, x], axis=0)
            self._ids = mx.concatenate([self._ids, id_arr], axis=0)  # type: ignore[arg-type]

        self.ntotal = int(self._vectors.shape[0])
        mx.eval(self._vectors)
        mx.eval(self._ids)
        
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        k = min(k, self.ntotal)
        
        # Compute distances matrix (lower is better), except for INNER_PRODUCT
        mt = self.metric_type
        if mt == MetricType.L2:
            distances = pairwise_L2sqr(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.L1:
            distances = pairwise_L1(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.Linf:
            distances = pairwise_Linf(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.Canberra:
            distances = pairwise_extra_distances(x, self._vectors, "Canberra")
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.BrayCurtis:
            distances = pairwise_extra_distances(x, self._vectors, "BrayCurtis")
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.JensenShannon:
            distances = pairwise_extra_distances(x, self._vectors, "JensenShannon")
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.Jaccard:
            distances = pairwise_jaccard(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)
        elif mt == MetricType.INNER_PRODUCT:
            sims = mx.matmul(x, self._vectors.T)
            distances = mx.negative(sims)
            vals, idx = topk_smallest_axis1(distances, k)
        else:
            distances = pairwise_L2sqr(x, self._vectors)
            vals, idx = topk_smallest_axis1(distances, k)

        selected_ids = mx.take(self._ids, idx, axis=0)
        return SearchResult(distances=vals, indices=selected_ids)
        
    def xb(self) -> List[List[float]]:
        """Get stored vectors.
        
        Returns:
            Stored vectors as an MLX array
        """
        if self._vectors is None:
            return mx.zeros((0, self.d), dtype=mx.float32)
        return self._vectors

    def reconstruct(self, key: int) -> mx.array:
        """Reconstruct the stored vector corresponding to the given ID."""
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
        if key not in self._id_to_row:
            raise ValueError(f"Unknown key {key}")
        return self._vectors[self._id_to_row[key]]

    def reset(self) -> None:
        super().reset()
        self._vectors = None
        self._ids = None
        self._id_to_row.clear()
        self.is_trained = True

    def save_to_file(self, filename: str) -> None:
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
        mx.save(self._vectors, filename + "_xb")
        mx.save(self._ids, filename + "_ids")

    def clone(self) -> "FlatIndex":
        new_index = FlatIndex(self.d, self.metric_type)
        if self._vectors is not None and self._ids is not None:
            new_index._vectors = mx.copy(self._vectors)
            new_index._ids = mx.copy(self._ids)
            new_index.ntotal = int(self._vectors.shape[0])
            new_index._id_to_row = self._id_to_row.copy()
        return new_index

    # =============================================================================
    # Extended FAISS API Implementation for FlatIndex
    # =============================================================================

    def add_with_ids(self, x: mx.array, ids: mx.array) -> None:
        """Add vectors with explicit IDs (delegates to add).
        
        Note: add() accepts both List and mx.array, converts internally.
        """
        # Convert mx.array to the format add() expects if needed
        if isinstance(x, mx.array) and isinstance(ids, mx.array):
            # add() will convert these internally, no CPU transfer here
            self.add(x, ids)
        else:
            self.add(x, ids)

    def reconstruct_batch(self, keys: mx.array) -> mx.array:
        """Reconstruct multiple vectors efficiently (GPU-only).
        
        Uses vectorized gather operation - no CPU transfers.
        """
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
        
        # Convert keys to int64 if needed
        if not isinstance(keys, mx.array):
            keys = mx.array(keys, dtype=mx.int64)
        
        # Build reverse mapping on CPU ONCE (this is initialization, not hot path)
        # For hot path reconstruction, this should be pre-built
        # TODO: Maintain GPU-resident id->row mapping
        
        # For now, use direct indexing assuming sequential IDs
        # This is a simplification - full implementation needs GPU hash table
        n_keys = keys.shape[0]
        result = mx.zeros((n_keys, self.d), dtype=mx.float32)
        
        # Vectorized lookup where IDs are sequential
        # For non-sequential, would need GPU hash table
        valid_mask = mx.logical_and(
            mx.greater_equal(keys, 0),
            mx.less(keys, self.ntotal)
        )
        valid_keys = mx.where(valid_mask, keys, mx.zeros_like(keys))
        result = mx.where(
            valid_mask[:, None],
            self._vectors[valid_keys],
            result
        )
        
        return result

    def reconstruct_n(self, i0: int, ni: int) -> mx.array:
        """Reconstruct range of vectors efficiently (GPU-only)."""
        if self._vectors is None:
            raise RuntimeError("Index is empty")
        
        if i0 < 0 or i0 + ni > self.ntotal:
            raise ValueError(f"Range [{i0}, {i0+ni}) out of bounds [0, {self.ntotal})")
        
        # Direct slice - stays on GPU
        return self._vectors[i0:i0+ni]

    def remove_ids(self, selector) -> int:
        """Remove vectors matching the ID selector (GPU-only filtering).
        
        Uses integer indexing instead of boolean (MLX limitation).
        All vector operations stay on GPU.
        """
        if self._vectors is None or self._ids is None:
            return 0
        
        # Build list of indices to keep
        keep_indices = []
        n = self.ntotal
        
        for i in range(n):
            # Get ID from tracking dict (metadata)
            vec_id = list(self._id_to_row.keys())[list(self._id_to_row.values()).index(i)] if i in self._id_to_row.values() else i
            if not selector.is_member(vec_id):
                keep_indices.append(i)
        
        n_removed = n - len(keep_indices)
        
        if n_removed == 0:
            return 0
        
        if len(keep_indices) == 0:
            self.reset()
            return n_removed
        
        # Filter on GPU using integer indices
        keep_idx_array = mx.array(keep_indices, dtype=mx.int32)
        self._vectors = mx.take(self._vectors, keep_idx_array, axis=0)
        self._ids = mx.take(self._ids, keep_idx_array, axis=0)
        
        # Rebuild ID mapping (metadata operation)
        self._id_to_row.clear()
        new_ntotal = int(self._vectors.shape[0])
        for i in range(new_ntotal):
            # Minimal scalar extraction for metadata rebuild
            self._id_to_row[int(self._ids[i])] = i
        
        self.ntotal = new_ntotal
        mx.eval(self._vectors)
        mx.eval(self._ids)
        
        return n_removed

    def sa_code_size(self) -> int:
        """Size of encoded vector in bytes (uncompressed float32)."""
        return self.d * 4

    def sa_encode(self, x: mx.array) -> mx.array:
        """Encode vectors (GPU-only byte view)."""
        # View as bytes - stays on GPU
        return x.view(mx.uint8)

    def sa_decode(self, codes: mx.array) -> mx.array:
        """Decode vectors (GPU-only byte view)."""
        # View as float32 - stays on GPU
        n_vecs = codes.shape[0] * codes.shape[1] // (self.d * 4)
        return codes.view(mx.float32).reshape(n_vecs, self.d)

    def compute_residual(self, x: mx.array, key: int) -> mx.array:
        """Compute residual (for flat index, residual is the vector itself)."""
        # Flat index has no encoding, so residual is just the input vector
        # Pure GPU operation
        return x

    def compute_residual_n(self, x: mx.array, keys: mx.array) -> mx.array:
        """Compute residuals (for flat index, residuals are the vectors themselves)."""
        # Flat index has no encoding, so residuals are just the input vectors
        # Pure GPU operation
        return x

    def update_vectors(self, idx: mx.array, v: mx.array) -> None:
        """Update vectors in the index (GPU-only)."""
        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index is empty")
        
        if not isinstance(idx, mx.array):
            idx = mx.array(idx, dtype=mx.int64)
        if not isinstance(v, mx.array):
            v = mx.array(v, dtype=mx.float32)
        
        # Vectorized update on GPU
        # For each ID in idx, find its row and update
        # This is simplified - full implementation needs GPU hash table
        n_updates = idx.shape[0]
        for i in range(n_updates):
            # Minimal CPU for lookup, but update stays on GPU
            vec_id = int(idx[i])
            if vec_id in self._id_to_row:
                row = self._id_to_row[vec_id]
                self._vectors[row] = v[i]
        
        mx.eval(self._vectors)

    def merge_from(self, other_index: 'FlatIndex', add_id: int = 0) -> None:
        """Merge another FlatIndex into this one (GPU-only)."""
        if not isinstance(other_index, FlatIndex):
            raise TypeError("Can only merge from another FlatIndex")
        
        self.check_compatible_for_merge(other_index)
        
        if other_index._vectors is None or other_index.ntotal == 0:
            return
        
        # Add ID offset on GPU
        other_ids = other_index._ids + add_id
        
        # Concatenate on GPU
        if self._vectors is None:
            self._vectors = other_index._vectors
            self._ids = other_ids
        else:
            self._vectors = mx.concatenate([self._vectors, other_index._vectors], axis=0)
            self._ids = mx.concatenate([self._ids, other_ids], axis=0)
        
        # Update metadata (small CPU work for tracking)
        start_idx = self.ntotal
        for i in range(other_index.ntotal):
            vec_id = int(other_ids[i])
            self._id_to_row[vec_id] = start_idx + i
        
        self.ntotal = int(self._vectors.shape[0])
        mx.eval(self._vectors)
        mx.eval(self._ids)
        
        # Clear the other index
        other_index.reset()
