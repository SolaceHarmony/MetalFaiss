"""
shards_index.py - Index sharding implementation (GPU/Metal-only)

Implements FAISS IndexShards functionality for horizontal scaling.
ALL OPERATIONS STAY ON GPU - NO CPU TRANSFERS.

Reference: faiss/IndexShards.{cpp,h}
Priority: P1 in IMPLEMENTATION_ROADMAP.md (Issue #1)
"""

from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from ..utils.search_result import SearchResult, SearchRangeResult
from ..types.metric_type import MetricType


class IndexShards(BaseIndex):
    """Index that shards vectors across multiple sub-indexes (GPU-only).
    
    Distributes vectors across N independent sub-indexes (shards) for
    horizontal scaling. ALL operations stay on Metal/GPU.
    
    FAISS Reference: faiss/IndexShards.h
    """
    
    def __init__(self, d: int, threaded: bool = False, successive: bool = True):
        """Initialize sharded index.
        
        Args:
            d: Vector dimension
            threaded: If True, use threading (MLX handles parallelism, ignored)
            successive: If True, add in round-robin (True), else all to first shard
        """
        super().__init__(d)
        self.shards: List[BaseIndex] = []
        self.threaded = threaded
        self.successive = successive
        self.own_fields = True
        self.is_trained = True
        
    def add_shard(self, index: BaseIndex) -> None:
        """Add a shard to the collection."""
        if index.d != self.d:
            raise ValueError(f"Shard dimension {index.d} != expected {self.d}")
        
        self.shards.append(index)
        self.ntotal = sum(s.ntotal for s in self.shards)
        self.is_trained = all(s.is_trained for s in self.shards)
        
    def add(self, xs, ids: Optional = None) -> None:
        """Add vectors, distributing across shards (GPU-only).
        
        Args:
            xs: Vectors to add (converted to mx.array internally)
            ids: Optional vector IDs (converted to mx.array internally)
        """
        if not self.shards:
            raise RuntimeError("No shards added to IndexShards")
        
        # Convert to mx.array if needed (input convenience, internal GPU-only)
        if not isinstance(xs, mx.array):
            x = mx.array(xs, dtype=mx.float32)
        else:
            x = xs
            
        if x.shape[1] != self.d:
            raise ValueError(f"Vector dimension {x.shape[1]} != {self.d}")
        
        n = x.shape[0]
        n_shards = len(self.shards)
        
        if self.successive:
            # Round-robin distribution on GPU
            for shard_idx in range(n_shards):
                # Create index mask for this shard on GPU
                indices = mx.arange(shard_idx, n, n_shards, dtype=mx.int32)
                if indices.shape[0] == 0:
                    continue
                
                # Gather vectors for this shard on GPU
                shard_vecs = x[indices]
                
                # Delegate to shard (shard handles conversion if needed)
                if ids is not None:
                    if not isinstance(ids, mx.array):
                        ids_arr = mx.array(ids, dtype=mx.int64)
                    else:
                        ids_arr = ids
                    shard_ids = ids_arr[indices]
                    self.shards[shard_idx].add(shard_vecs, shard_ids)
                else:
                    self.shards[shard_idx].add(shard_vecs)
        else:
            # Add all to first shard
            if ids is not None:
                self.shards[0].add(x, ids)
            else:
                self.shards[0].add(x)
        
        self.ntotal = sum(s.ntotal for s in self.shards)
    
    def add_with_ids(self, xs, ids) -> None:
        """Add vectors with explicit IDs."""
        self.add(xs, ids)
        
    def search(self, xs, k: int) -> SearchResult:
        """Search all shards and merge top-k results (GPU-only).
        
        All merging happens on GPU - no CPU transfers.
        """
        if not self.shards:
            raise RuntimeError("No shards added to IndexShards")
        
        # Convert to mx.array if needed
        if not isinstance(xs, mx.array):
            x = mx.array(xs, dtype=mx.float32)
        else:
            x = xs
            
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} != {self.d}")
        
        nq = x.shape[0]
        
        # Search each shard - results stay on GPU
        all_distances = []
        all_labels = []
        
        for shard in self.shards:
            if shard.ntotal == 0:
                continue
            
            result = shard.search(x, min(k, shard.ntotal))
            all_distances.append(result.distances)
            all_labels.append(result.indices)
        
        if not all_distances:
            # No vectors in any shard
            return SearchResult(
                distances=mx.full((nq, k), float('inf'), dtype=mx.float32),
                indices=mx.full((nq, k), -1, dtype=mx.int64)
            )
        
        # Concatenate results on GPU
        distances = mx.concatenate(all_distances, axis=1)
        labels = mx.concatenate(all_labels, axis=1)
        
        # Merge top-k on GPU using argsort
        # For each query, find k smallest distances
        sorted_indices = mx.argsort(distances, axis=1)[:, :k]
        
        # Gather top-k distances and labels on GPU
        batch_indices = mx.arange(nq, dtype=mx.int32)[:, None]
        final_distances = distances[batch_indices, sorted_indices]
        final_labels = labels[batch_indices, sorted_indices]
        
        # Pad if needed (on GPU)
        actual_k = final_distances.shape[1]
        if actual_k < k:
            pad_size = k - actual_k
            final_distances = mx.concatenate([
                final_distances,
                mx.full((nq, pad_size), float('inf'), dtype=mx.float32)
            ], axis=1)
            final_labels = mx.concatenate([
                final_labels,
                mx.full((nq, pad_size), -1, dtype=mx.int64)
            ], axis=1)
        
        return SearchResult(distances=final_distances, indices=final_labels)
    
    def train(self, xs) -> None:
        """Train all shards that need training."""
        # Convert to mx.array if needed
        if not isinstance(xs, mx.array):
            xs = mx.array(xs, dtype=mx.float32)
            
        for shard in self.shards:
            if not shard.is_trained:
                shard.train(xs)
        
        self.is_trained = all(s.is_trained for s in self.shards)
    
    def reset(self) -> None:
        """Reset all shards."""
        for shard in self.shards:
            shard.reset()
        self.ntotal = 0
        
    def remove_ids(self, selector) -> int:
        """Remove IDs from all shards."""
        total_removed = 0
        for shard in self.shards:
            total_removed += shard.remove_ids(selector)
        
        self.ntotal = sum(s.ntotal for s in self.shards)
        return total_removed
    
    @property
    def metric_type(self) -> MetricType:
        """Get metric type from first shard."""
        if self.shards:
            return self.shards[0].metric_type
        return MetricType.L2
    
    def __repr__(self) -> str:
        return f"IndexShards(d={self.d}, n_shards={len(self.shards)}, ntotal={self.ntotal})"
