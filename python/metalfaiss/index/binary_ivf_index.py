"""
binary_ivf_index.py - Base class for binary IVF (Inverted File) indexes

This implements inverted file indexing for binary vectors, using Hamming
distance for both coarse quantization and distance computation.

Original: faiss/IndexBinaryIVF.h
"""

import mlx.core as mx
from typing import List, Optional, Dict, Tuple
from .binary_index import BaseBinaryIndex
from .binary_flat_index import BinaryFlatIndex
from ..utils.search_result import SearchResult
from ..errors import InvalidArgumentError

class BinaryIVFIndex(BaseBinaryIndex):
    """Base class for binary IVF (Inverted File) indexes.
    
    This partitions binary vectors into inverted lists using a coarse
    quantizer that operates in Hamming space. Each list contains vectors
    that are close to the same centroid in Hamming distance.
    """
    
    def __init__(self, quantizer: BinaryFlatIndex, d: int, nlist: int):
        """Initialize binary IVF index.
        
        Args:
            quantizer: Coarse quantizer for binary vectors
            d: Dimension in bits
            nlist: Number of inverted lists (partitions)
            
        Raises:
            InvalidArgumentError: If dimensions don't match or nlist invalid
        """
        if quantizer.d != d:
            raise InvalidArgumentError(
                f"Quantizer dimension {quantizer.d} does not match index dimension {d}"
            )
            
        if nlist <= 0:
            raise InvalidArgumentError("Number of lists must be positive")
            
        super().__init__(d)
        self._quantizer = quantizer
        self._nlist = nlist
        self._nprobe = 1  # Number of lists to probe during search
        
        # Inverted lists: list index -> [(id, vector), ...]
        self._invlists: List[List[Tuple[int, mx.array]]] = [
            [] for _ in range(nlist)
        ]
        
    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist
        
    @property
    def nprobe(self) -> int:
        """Number of lists to probe during search."""
        return self._nprobe
        
    @nprobe.setter
    def nprobe(self, value: int) -> None:
        """Set number of lists to probe.
        
        Args:
            value: Number of lists (must be positive)
            
        Raises:
            ValueError: If value not positive
        """
        if value < 1:
            raise ValueError("nprobe must be positive")
        self._nprobe = value
        
    @property
    def quantizer(self) -> BinaryFlatIndex:
        """The coarse quantizer used by this index."""
        return self._quantizer
        
    def _train(self, xs: List[List[int]]) -> None:
        """Train the index.
        
        This trains the coarse quantizer on the binary training vectors.
        
        Args:
            xs: Training vectors (each component must be 0 or 1)
        """
        if not xs:
            raise ValueError("Empty training data")
            
        # Train quantizer
        self._quantizer.train(xs)
        
    def _add(
        self,
        xs: List[List[int]],
        ids: Optional[List[int]] = None
    ) -> None:
        """Add binary vectors to the index.
        
        Args:
            xs: Vectors to add (each component must be 0 or 1)
            ids: Optional vector IDs
        """
        x = mx.array(xs, dtype=mx.uint8)
        
        # Assign vectors to lists using quantizer
        assignments = self._quantizer.search(xs, 1)
        for i, label in enumerate(assignments.labels):
            self._invlists[label[0]].append(
                (ids[i] if ids else i, x[i])
            )
            
        self._ntotal += len(x)
        
    def _search(self, xs: List[List[int]], k: int) -> SearchResult:
        """Search for nearest neighbors by Hamming distance.
        
        This performs coarse quantization to identify candidate lists,
        then computes exact Hamming distances to vectors in those lists.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if self._ntotal == 0:
            raise RuntimeError("Index is empty")
            
        x = mx.array(xs, dtype=mx.uint8)
        n = len(x)
        
        # Find nprobe closest centroids for each query
        coarse = self._quantizer.search(xs, self._nprobe)
        
        # Search each relevant list
        all_distances = []  # type: List[List[int]]
        all_labels = []  # type: List[List[int]]
        
        for i in range(n):  # For each query
            # Collect candidates from relevant lists
            candidates = []  # type: List[Tuple[int, mx.array]]
            for j in range(self._nprobe):
                list_no = coarse.labels[i][j]
                candidates.extend(self._invlists[list_no])
                
            if not candidates:
                # No candidates found
                all_distances.append([float('inf')] * k)
                all_labels.append([-1] * k)
                continue
                
            # Compute Hamming distances to candidates
            cand_vecs = mx.stack([v for _, v in candidates])
            cand_ids = [id for id, _ in candidates]
            
            dists = self.hamming_distances(x[i:i+1], cand_vecs)[0]
            
            # Get top k
            if len(dists) <= k:
                # Not enough candidates
                sorted_idx = mx.argsort(dists)
                distances = dists[sorted_idx].tolist()
                labels = [cand_ids[j] for j in sorted_idx.tolist()]
                
                # Pad with sentinel values
                distances.extend([float('inf')] * (k - len(distances)))
                labels.extend([-1] * (k - len(labels)))
            else:
                # Get top k
                values, indices = mx.topk(-dists, k)
                distances = (-values).tolist()
                labels = [cand_ids[j] for j in indices.tolist()]
                
            all_distances.append(distances)
            all_labels.append(labels)
            
        return SearchResult(
            distances=all_distances,
            labels=all_labels
        )
        
    def _reconstruct(self, key: int) -> List[int]:
        """Reconstruct vector from storage.
        
        Args:
            key: Vector ID to reconstruct
            
        Returns:
            Reconstructed vector
            
        Raises:
            RuntimeError: If index empty
            ValueError: If key invalid
        """
        if self._ntotal == 0:
            raise RuntimeError("Index is empty")
            
        # Search all lists for the key
        for invlist in self._invlists:
            for id, vec in invlist:
                if id == key:
                    return vec.tolist()
                    
        raise ValueError(f"Invalid key {key}")
        
    def reset(self) -> None:
        """Reset the index."""
        super().reset()
        self._invlists = [[] for _ in range(self._nlist)]
        self._quantizer.reset()