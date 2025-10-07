"""
replicas_index.py - Index replication implementation (GPU/Metal-only)

Implements FAISS IndexReplicas functionality for load balancing.
ALL OPERATIONS STAY ON GPU - NO CPU TRANSFERS.

Reference: faiss/IndexReplicas.{cpp,h}
Priority: P1 in IMPLEMENTATION_ROADMAP.md (Issue #2)
"""

from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from ..utils.search_result import SearchResult, SearchRangeResult
from ..types.metric_type import MetricType


class IndexReplicas(BaseIndex):
    """Index that maintains multiple replicas for load balancing (GPU-only).
    
    Mirrors vectors across N independent replica indexes.
    ALL operations stay on Metal/GPU.
    
    FAISS Reference: faiss/IndexReplicas.h
    """
    
    def __init__(self, d: int, threaded: bool = False):
        """Initialize replicated index.
        
        Args:
            d: Vector dimension
            threaded: If True, use threading (MLX handles parallelism, ignored)
        """
        super().__init__(d)
        self.replicas: List[BaseIndex] = []
        self.threaded = threaded
        self.own_fields = True
        self.is_trained = True
        self._search_replica_idx = 0
        
    def add_replica(self, index: BaseIndex) -> None:
        """Add a replica to the collection."""
        if index.d != self.d:
            raise ValueError(f"Replica dimension {index.d} != expected {self.d}")
        
        self.replicas.append(index)
        
        if self.replicas:
            self.ntotal = self.replicas[0].ntotal
        
        self.is_trained = all(r.is_trained for r in self.replicas)
        
    def add(self, xs, ids: Optional = None) -> None:
        """Add vectors, mirroring across all replicas (GPU-only).
        
        Args:
            xs: Vectors to add (converted to mx.array internally)
            ids: Optional vector IDs (converted to mx.array internally)
        """
        if not self.replicas:
            raise RuntimeError("No replicas added to IndexReplicas")
        
        # Convert to mx.array if needed (input convenience, internal GPU-only)
        if not isinstance(xs, mx.array):
            x = mx.array(xs, dtype=mx.float32)
        else:
            x = xs
        
        # Convert ids if needed
        if ids is not None and not isinstance(ids, mx.array):
            ids = mx.array(ids, dtype=mx.int64)
        
        # Mirror add to all replicas (data stays on GPU)
        for replica in self.replicas:
            if ids is not None:
                replica.add(x, ids)
            else:
                replica.add(x)
        
        self.ntotal = self.replicas[0].ntotal if self.replicas else 0
    
    def add_with_ids(self, xs, ids) -> None:
        """Add vectors with explicit IDs, mirroring to all replicas."""
        self.add(xs, ids)
        
    def search(
        self, 
        xs, 
        k: int, 
        use_all_replicas: bool = False
    ) -> SearchResult:
        """Search replicas (GPU-only).
        
        Args:
            xs: Query vectors
            k: Number of neighbors
            use_all_replicas: If True, search all replicas and merge results
                             If False, search one replica (round-robin)
        
        All operations stay on GPU.
        """
        if not self.replicas:
            raise RuntimeError("No replicas added to IndexReplicas")
        
        # Convert to mx.array if needed
        if not isinstance(xs, mx.array):
            x = mx.array(xs, dtype=mx.float32)
        else:
            x = xs
        
        if not use_all_replicas:
            # Round-robin: search one replica
            replica = self.replicas[self._search_replica_idx]
            self._search_replica_idx = (self._search_replica_idx + 1) % len(self.replicas)
            return replica.search(x, k)
        else:
            # Search all replicas and merge results on GPU
            nq = x.shape[0]
            
            # Search each replica - results stay on GPU
            all_distances = []
            all_labels = []
            
            for replica in self.replicas:
                result = replica.search(x, k)
                all_distances.append(result.distances)
                all_labels.append(result.indices)
            
            if not all_distances:
                return SearchResult(
                    distances=mx.full((nq, k), float('inf'), dtype=mx.float32),
                    indices=mx.full((nq, k), -1, dtype=mx.int64)
                )
            
            # Concatenate results on GPU
            distances = mx.concatenate(all_distances, axis=1)
            labels = mx.concatenate(all_labels, axis=1)
            
            # Merge top-k on GPU using argsort
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
        """Train all replicas that need training."""
        # Convert to mx.array if needed
        if not isinstance(xs, mx.array):
            xs = mx.array(xs, dtype=mx.float32)
            
        for replica in self.replicas:
            if not replica.is_trained:
                replica.train(xs)
        
        self.is_trained = all(r.is_trained for r in self.replicas)
    
    def reset(self) -> None:
        """Reset all replicas."""
        for replica in self.replicas:
            replica.reset()
        self.ntotal = 0
        
    def remove_ids(self, selector) -> int:
        """Remove IDs from all replicas."""
        if not self.replicas:
            return 0
        
        # Remove from first replica
        n_removed = self.replicas[0].remove_ids(selector)
        
        # Remove from remaining replicas
        for replica in self.replicas[1:]:
            replica.remove_ids(selector)
        
        self.ntotal = self.replicas[0].ntotal if self.replicas else 0
        return n_removed
    
    @property
    def metric_type(self) -> MetricType:
        """Get metric type from first replica."""
        if self.replicas:
            return self.replicas[0].metric_type
        return MetricType.L2
    
    def __repr__(self) -> str:
        return f"IndexReplicas(d={self.d}, n_replicas={len(self.replicas)}, ntotal={self.ntotal})"
