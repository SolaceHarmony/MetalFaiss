# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

# TODO: Review implementations from:
# ✓ faiss/impl/IDSelector.h (base selector interfaces)
# - faiss/impl/IDSelector.cpp (selector implementations)
# - faiss/MetaIndexes.h (selector usage examples)
# - faiss/impl/FaissException.h (error handling)
# - faiss/impl/platform_macros.h (platform-specific optimizations)

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Set, List  # Added List import

class IDSelector(ABC):
    """Base class for selecting subsets of vectors."""
    
    @abstractmethod
    def is_member(self, id: int) -> bool:
        """Check if ID is selected."""
        pass

class IDSelectorRange(IDSelector):
    """Select IDs in range [imin, imax)."""
    
    def __init__(self, imin: int, imax: int, assume_sorted: bool = False):
        self.imin = imin
        self.imax = imax
        self.assume_sorted = assume_sorted
        
    def is_member(self, id: int) -> bool:
        return self.imin <= id < self.imax

    def find_sorted_ids_bounds(
        self,
        list_size: int,
        ids: np.ndarray,
        jmin: List[int],
        jmax: List[int]
    ) -> None:
        """Find range bounds in sorted list."""
        if not self.assume_sorted:
            jmin[0] = 0
            jmax[0] = list_size
            return
            
        # Binary search for bounds
        jmin[0] = np.searchsorted(ids[:list_size], self.imin, side='left')
        jmax[0] = np.searchsorted(ids[:list_size], self.imax, side='right')

class IDSelectorArray(IDSelector):
    """Select IDs from array."""
    
    def __init__(self, ids: np.ndarray):
        self.ids = set(ids)
        
    def is_member(self, id: int) -> bool:
        return id in self.ids

class IDSelectorBatch(IDSelector):
    """Select IDs with Bloom filter optimization."""
    
    def __init__(self, indices: np.ndarray):
        # Set based lookup
        self.set = set(indices)
        
        # Bloom filter parameters
        self.nbits = max(64, len(indices) * 2)
        self.mask = (1 << int(np.log2(self.nbits))) - 1
        nbytes = (self.nbits + 7) // 8
        self.bloom = np.zeros(nbytes, dtype=np.uint8)
        
        # Build Bloom filter
        for idx in indices:
            h = hash(idx) & self.mask  # Simple hash
            self.bloom[h >> 3] |= 1 << (h & 7)
            
    def is_member(self, id: int) -> bool:
        # Check Bloom filter first
        h = hash(id) & self.mask
        if not (self.bloom[h >> 3] & (1 << (h & 7))):
            return False
        # Full check only if Bloom filter passes
        return id in self.set

class IDSelectorBitmap(IDSelector):
    """Select IDs using bitmap."""
    
    def __init__(self, n: int, bitmap: np.ndarray):
        self.n = n
        self.bitmap = bitmap
        
    def is_member(self, id: int) -> bool:
        byte_idx = id >> 3
        if byte_idx >= self.n:
            return False
        return bool(self.bitmap[byte_idx] & (1 << (id & 7)))

class IDSelectorNot(IDSelector):
    """Invert another selector."""
    
    def __init__(self, sel: IDSelector):
        self.sel = sel
        
    def is_member(self, id: int) -> bool:
        return not self.sel.is_member(id)

class IDSelectorAll(IDSelector):
    """Select all IDs."""
    
    def is_member(self, id: int) -> bool:
        return True

class IDSelectorAnd(IDSelector):
    """Combine two selectors with AND."""
    
    def __init__(self, lhs: IDSelector, rhs: IDSelector):
        self.lhs = lhs
        self.rhs = rhs
        
    def is_member(self, id: int) -> bool:
        return self.lhs.is_member(id) and self.rhs.is_member(id)

class IDSelectorOr(IDSelector):
    """Combine two selectors with OR."""
    
    def __init__(self, lhs: IDSelector, rhs: IDSelector):
        self.lhs = lhs
        self.rhs = rhs  # Fixed missing assignment
        
    def is_member(self, id: int) -> bool:
        return self.lhs.is_member(id) or self.rhs.is_member(id)

class IDSelectorXOr(IDSelector):
    """Combine two selectors with XOR."""
    
    def __init__(self, lhs: IDSelector, rhs: IDSelector):
        self.lhs = lhs
        self.rhs = rhs  # Fixed missing assignment
        
    def is_member(self, id: int) -> bool:
        return self.lhs.is_member(id) ^ self.rhs.is_member(id)
