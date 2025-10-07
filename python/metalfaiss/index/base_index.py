"""
base_index.py - Base class for all indices
"""

from ..utils.search_result import SearchResult, SearchRangeResult
from ..types.metric_type import MetricType
from typing import Optional, List, Tuple, Union
import mlx.core as mx

class BaseIndex:
    """Base class for all indices."""
    
    def __init__(self, d: int, metric: MetricType = MetricType.L2):
        """Initialize base index.
        
        Args:
            d: Dimension of vectors
            metric: Distance metric to use
        """
        self.d = d
        self.metric = metric
        self._metric_type = metric
        self.is_trained = False
        self.ntotal = 0
        
    def train(self, x: mx.array) -> None:
        """Train the index.
        
        Args:
            x: Training vectors (n, d)
        """
        if x.shape[1] != self.d:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != index dimension {self.d}")
        self.is_trained = True
        
    def add(self, x: mx.array) -> None:
        """Add vectors to the index.
        
        Args:
            x: Vectors to add (n, d)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        if x.shape[1] != self.d:
            raise ValueError(f"Vector dimension {x.shape[1]} != index dimension {self.d}")
        self.ntotal += x.shape[0]
        
    def search(self, x: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        """Search for nearest neighbors.
        
        Args:
            x: Query vectors (n, d)
            k: Number of nearest neighbors
            
        Returns:
            distances: Distances to nearest neighbors (n, k)
            indices: Indices of nearest neighbors (n, k)
        """
        raise NotImplementedError
        
    def range_search(self, x: mx.array, radius: float) -> SearchRangeResult:
        """Search for vectors within radius.
        
        Args:
            x: Query vectors (n, d)
            radius: Search radius
            
        Returns:
            SearchRangeResult containing distances and indices
        """
        raise NotImplementedError
        
    def reset(self) -> None:
        """Reset the index."""
        self.ntotal = 0
        self.is_trained = False

    # GPU-only project: keep a no-op `.to_gpu` for compatibility
    def to_gpu(self, resources=None):  # type: ignore[override]
        return self

    # =============================================================================
    # Extended FAISS API - Methods below match C++ FAISS Index interface
    # =============================================================================

    def add_with_ids(self, x: mx.array, ids: mx.array) -> None:
        """Add vectors with explicit IDs.
        
        FAISS API: void add_with_ids(idx_t n, const float* x, const idx_t* xids)
        
        Args:
            x: Vectors to add, shape (n, d)
            ids: Explicit IDs for vectors, shape (n,)
        
        TODO: Implement in subclasses that support explicit IDs (IDMap, IVF, etc.)
        Reference: faiss/Index.h line ~100
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.add_with_ids() not implemented. "
            "See IMPLEMENTATION_ROADMAP.md or wrap with IDMap"
        )

    def assign(self, x: mx.array, k: int = 1) -> mx.array:
        """Return only the labels (no distances) for k nearest neighbors.
        
        FAISS API: void assign(idx_t n, const float* x, idx_t* labels, idx_t k)
        
        Args:
            x: Query vectors, shape (n, d)
            k: Number of neighbors
            
        Returns:
            labels: Neighbor IDs, shape (n, k)
        
        TODO: Optimize - currently just calls search() and discards distances
        Reference: faiss/Index.cpp
        """
        _, labels = self.search(x, k)
        return labels

    def reconstruct(self, key: int) -> mx.array:
        """Reconstruct a single vector from the index.
        
        FAISS API: void reconstruct(idx_t key, float* recons)
        
        Args:
            key: Index of vector to reconstruct
            
        Returns:
            Vector of shape (d,)
        
        TODO: Implement in subclasses (Flat, IVF, PQ, etc.)
        Reference: faiss/Index.h line ~175
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.reconstruct() not implemented"
        )

    def reconstruct_batch(self, keys: mx.array) -> mx.array:
        """Reconstruct multiple vectors from the index.
        
        FAISS API: void reconstruct_batch(idx_t n, const idx_t* keys, float* recons)
        
        Args:
            keys: Vector indices to reconstruct, shape (n,)
            
        Returns:
            Reconstructed vectors, shape (n, d)
        
        TODO: Implement batch reconstruction (more efficient than loop)
        Reference: faiss/Index.h line ~186
        
        Note: Default implementation uses reconstruct() in loop.
        Subclasses should override with vectorized GPU implementation.
        """
        # Default implementation: loop over reconstruct()
        # This is fallback only - subclasses should provide GPU-vectorized version
        if not isinstance(keys, mx.array):
            keys = mx.array(keys, dtype=mx.int64)
        
        n = keys.shape[0]
        vecs = []
        for i in range(n):
            # Note: This loop is CPU-bound metadata access
            # Subclasses should override with pure GPU gather operations
            vecs.append(self.reconstruct(int(keys[i])))
        
        return mx.stack(vecs) if vecs else mx.zeros((0, self.d))

    def reconstruct_n(self, i0: int, ni: int) -> mx.array:
        """Reconstruct a range of vectors [i0, i0+ni).
        
        FAISS API: void reconstruct_n(idx_t i0, idx_t ni, float* recons)
        
        Args:
            i0: Starting index
            ni: Number of vectors to reconstruct
            
        Returns:
            Reconstructed vectors, shape (ni, d)
        
        TODO: Implement efficient range reconstruction in subclasses
        Reference: faiss/Index.h line ~195
        Priority: P2 in IMPLEMENTATION_ROADMAP.md (Issue #7)
        
        Note: Default uses reconstruct_batch with arange on GPU.
        """
        keys = mx.arange(i0, i0 + ni, dtype=mx.int32)
        return self.reconstruct_batch(keys)

    def search_and_reconstruct(
        self, x: mx.array, k: int
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Search and return both neighbors and reconstructed vectors.
        
        FAISS API: void search_and_reconstruct(idx_t n, const float* x, idx_t k,
                                               float* distances, idx_t* labels, 
                                               float* recons)
        
        Args:
            x: Query vectors, shape (n, d)
            k: Number of neighbors
            
        Returns:
            distances: shape (n, k)
            labels: shape (n, k)
            recons: Reconstructed neighbor vectors, shape (n, k, d)
        
        TODO: Optimize in subclasses to avoid direct_map requirement
        Reference: faiss/Index.h line ~204, faiss/IndexIVF.cpp line ~800
        
        Note: Default implementation uses search() + reconstruct_batch().
        Subclasses should provide optimized GPU-only versions.
        """
        distances, labels = self.search(x, k)
        n = x.shape[0]
        recons = mx.zeros((n, k, self.d), dtype=mx.float32)
        
        # Reconstruct each neighbor using vectorized operations where possible
        # This is fallback - subclasses should override with pure GPU gather
        for i in range(n):
            # Get valid labels for this query (skip -1)
            query_labels = labels[i]
            valid_mask = mx.greater_equal(query_labels, 0)
            valid_labels = mx.where(valid_mask, query_labels, mx.zeros_like(query_labels))
            
            # Batch reconstruct valid neighbors
            try:
                query_recons = self.reconstruct_batch(valid_labels)
                recons[i] = mx.where(valid_mask[:, None], query_recons, recons[i])
            except NotImplementedError:
                # Fallback if reconstruct not implemented
                pass
        
        return distances, labels, recons

    def remove_ids(self, selector) -> int:
        """Remove vectors matching the ID selector.
        
        FAISS API: size_t remove_ids(const IDSelector& sel)
        
        Args:
            selector: IDSelector object (IDSelectorRange, IDSelectorBatch, etc.)
            
        Returns:
            Number of vectors removed
        
        TODO: Implement ID removal with inverted list consistency
        Reference: faiss/Index.cpp, faiss/IndexIVF.cpp line ~1100
        Priority: P2 in IMPLEMENTATION_ROADMAP.md (Issue #6)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.remove_ids() not implemented. "
            "See IMPLEMENTATION_ROADMAP.md Issue #6"
        )

    def compute_residual(self, x: mx.array, key: int) -> mx.array:
        """Compute residual vector after encoding.
        
        FAISS API: void compute_residual(const float* x, float* residual, idx_t key)
        
        The residual is the difference between the input vector and its
        reconstruction from the index.
        
        Args:
            x: Input vector, shape (d,)
            key: Encoding key (e.g., centroid ID for IVF)
            
        Returns:
            Residual vector, shape (d,)
        
        TODO: Implement for quantizer-based indexes (IVF, PQ)
        Reference: faiss/Index.h line ~219
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.compute_residual() not implemented"
        )

    def compute_residual_n(self, x: mx.array, keys: mx.array) -> mx.array:
        """Compute residuals for multiple vectors (batch form).
        
        FAISS API: void compute_residual_n(idx_t n, const float* xs,
                                          float* residuals, const idx_t* keys)
        
        Args:
            x: Input vectors, shape (n, d)
            keys: Encoding keys, shape (n,)
            
        Returns:
            Residual vectors, shape (n, d)
        
        TODO: Batch residual computation for IVF/quantizer indexes
        Reference: faiss/Index.h line ~232
        
        Note: Default implementation loops. Subclasses should provide
        vectorized GPU implementation.
        """
        n = x.shape[0]
        residuals = mx.zeros_like(x)
        
        # This loop is for fallback only
        # Subclasses should override with vectorized GPU operations
        for i in range(n):
            residuals[i] = self.compute_residual(x[i], int(keys[i]))
        
        return residuals

    def update_vectors(self, idx: mx.array, v: mx.array) -> None:
        """Update vectors in the index.
        
        FAISS API: void update_vectors(int nv, const idx_t* idx, const float* v)
        
        Requires direct_map for IVF indexes.
        
        Args:
            idx: Indices of vectors to update, shape (nv,)
            v: New vector values, shape (nv, d)
        
        TODO: Implement for indexes with direct_map (IVF)
        Reference: faiss/IndexIVF.cpp line ~950
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.update_vectors() not implemented. "
            "Requires direct_map support"
        )

    def merge_from(self, other_index: 'BaseIndex', add_id: int = 0) -> None:
        """Merge another index into this one.
        
        FAISS API: void merge_from(Index& otherIndex, idx_t add_id)
        
        Moves vectors from other_index to self. Other index is emptied.
        add_id is added to all transferred IDs.
        
        Args:
            other_index: Index to merge from (will be reset)
            add_id: Offset to add to IDs
        
        TODO: Implement index merging with ID offset handling
        Reference: faiss/Index.h line ~257
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.merge_from() not implemented"
        )

    def check_compatible_for_merge(self, other_index: 'BaseIndex') -> None:
        """Check if two indexes can be merged.
        
        FAISS API: void check_compatible_for_merge(const Index& otherIndex)
        
        Raises:
            ValueError: If indexes are incompatible
        
        TODO: Implement compatibility checks (dimension, metric, training)
        Reference: faiss/Index.h line ~262
        """
        if self.d != other_index.d:
            raise ValueError(
                f"Dimension mismatch: {self.d} vs {other_index.d}"
            )
        if self.metric_type != other_index.metric_type:
            raise ValueError(
                f"Metric mismatch: {self.metric_type} vs {other_index.metric_type}"
            )

    # =============================================================================
    # Standalone Codec Interface (SA) - for serialization/external storage
    # =============================================================================

    def sa_code_size(self) -> int:
        """Size of encoded vector in bytes.
        
        FAISS API: size_t sa_code_size()
        
        Returns:
            Code size in bytes per vector
        
        TODO: Implement for quantized indexes (PQ, SQ, IVF)
        Reference: faiss/Index.h line ~268
        """
        # Default: uncompressed float32
        return self.d * 4

    def sa_encode(self, x: mx.array) -> mx.array:
        """Encode vectors using standalone codec.
        
        FAISS API: void sa_encode(idx_t n, const float* x, uint8_t* bytes)
        
        Args:
            x: Vectors to encode, shape (n, d)
            
        Returns:
            Encoded bytes, shape (n, sa_code_size())
        
        TODO: Implement encoding for quantized indexes
        Reference: faiss/Index.h line ~276
        """
        # Default: cast to bytes
        n = x.shape[0]
        code_size = self.sa_code_size()
        return x.astype(mx.uint8).reshape(n, code_size)

    def sa_decode(self, codes: mx.array) -> mx.array:
        """Decode vectors from standalone codec.
        
        FAISS API: void sa_decode(idx_t n, const uint8_t* bytes, float* x)
        
        Args:
            codes: Encoded bytes, shape (n, sa_code_size())
            
        Returns:
            Decoded vectors, shape (n, d)
        
        TODO: Implement decoding for quantized indexes
        Reference: faiss/Index.h line ~284
        """
        # Default: cast from bytes
        return codes.astype(mx.float32).reshape(-1, self.d)

    def add_sa_codes(self, codes: mx.array, xids: mx.array) -> None:
        """Add pre-encoded vectors.
        
        FAISS API: void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids)
        
        Args:
            codes: Pre-encoded vectors, shape (n, sa_code_size())
            xids: Vector IDs, shape (n,)
        
        TODO: Implement for indexes supporting standalone codec
        Reference: faiss/Index.h line ~290
        """
        # Default: decode and add
        x = self.sa_decode(codes)
        self.add_with_ids(x, xids)
        
    def __len__(self) -> int:
        """Get number of vectors in index."""
        return self.ntotal

    @property
    def metric_type(self) -> MetricType:
        """Return the configured distance metric."""
        return self._metric_type

    @metric_type.setter
    def metric_type(self, value: MetricType) -> None:
        self._metric_type = value
