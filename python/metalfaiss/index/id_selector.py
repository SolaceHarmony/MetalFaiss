"""
id_selector.py - ID selection and filtering

This implements the IDSelector functionality from FAISS, used to define subsets
of vectors to handle during search or removal operations.

Original: faiss/impl/IDSelector.h
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional
from dataclasses import dataclass

class IDSelector(ABC):
    """Base class for ID selection."""
    
    @abstractmethod
    def is_member(self, id: int) -> bool:
        """Check if ID is selected."""
        pass

class IDSelectorRange(IDSelector):
    """Select IDs in range [imin, imax)."""
    
    def __init__(self, imin: int, imax: int, assume_sorted: bool = False):
        """Initialize range selector.
        
        Args:
            imin: Minimum ID (inclusive)
            imax: Maximum ID (exclusive)
            assume_sorted: Whether IDs are known to be sorted
        """
        self.imin = imin
        self.imax = imax
        self.assume_sorted = assume_sorted
        
    def is_member(self, id: int) -> bool:
        """Check if ID is in range."""
        return self.imin <= id < self.imax
        
    def find_sorted_ids_bounds(
        self,
        list_size: int,
        ids: np.ndarray
    ) -> tuple[int, int]:
        """Find bounds of valid IDs in sorted list.
        
        Args:
            list_size: Size of ID list
            ids: Array of sorted IDs
            
        Returns:
            Tuple of (min_idx, max_idx) for valid ID range
        """
        if not self.assume_sorted:
            raise ValueError("IDs must be sorted when assume_sorted=True")
            
        # Binary search for bounds
        jmin = np.searchsorted(ids[:list_size], self.imin, side='left')
        jmax = np.searchsorted(ids[:list_size], self.imax, side='right')
        
        return int(jmin), int(jmax)

class IDSelectorArray(IDSelector):
    """Select IDs from explicit array."""
    
    def __init__(self, ids: List[int]):
        """Initialize with array of IDs.
        
        Args:
            ids: List of IDs to select
        """
        self.ids = set(ids)  # Convert to set for O(1) lookup
        
    def is_member(self, id: int) -> bool:
        """Check if ID is in array."""
        return id in self.ids

class IDSelectorBatch(IDSelector):
    """Select IDs from batch with Bloom filter optimization."""
    
    def __init__(self, ids: List[int]):
        """Initialize with batch of IDs.
        
        Args:
            ids: List of IDs to select
        """
        self.id_set = set(ids)
        
        # Initialize Bloom filter
        # Use 8 bits per ID for good false positive rate
        self.nbits = max(64, 8 * len(ids))
        self.mask = (1 << self.nbits) - 1
        self.bloom = np.zeros(self.nbits // 8, dtype=np.uint8)
        
        # Add IDs to Bloom filter
        for id in ids:
            h = id & self.mask  # Simple hash function
            self.bloom[h >> 3] |= 1 << (h & 7)
            
    def is_member(self, id: int) -> bool:
        """Check if ID is selected using Bloom filter."""
        # Check Bloom filter first
        h = id & self.mask
        if not (self.bloom[h >> 3] & (1 << (h & 7))):
            return False  # Definitely not in set
            
        # Possible match, check actual set
        return id in self.id_set

class IDSelectorBitmap(IDSelector):
    """Select IDs using bitmap."""
    
    def __init__(self, n: int, bitmap: np.ndarray):
        """Initialize with bitmap.
        
        Args:
            n: Number of IDs
            bitmap: Bitmap array of ceil(n/8) bytes
        """
        self.n = n
        self.bitmap = bitmap
        
    def is_member(self, id: int) -> bool:
        """Check if ID is selected in bitmap."""
        if not (0 <= id < self.n):
            return False
        byte_idx = id >> 3  # Divide by 8
        bit_idx = id & 7    # Modulo 8
        return bool(self.bitmap[byte_idx] & (1 << bit_idx))

class IDSelectorAll(IDSelector):
    """Select all IDs (for benchmarking)."""
    
    def is_member(self, id: int) -> bool:
        """Always return True."""
        return True

class IDSelectorNot(IDSelector):
    """Invert another selector."""
    
    def __init__(self, selector: IDSelector):
        """Initialize with selector to invert.
        
        Args:
            selector: Selector to invert
        """
        self.selector = selector
        
    def is_member(self, id: int) -> bool:
        """Return inverse of wrapped selector."""
        return not self.selector.is_member(id)

class IDSelectorAnd(IDSelector):
    """Combine two selectors with AND."""
    
    def __init__(self, lhs: IDSelector, rhs: IDSelector):
        """Initialize with two selectors.
        
        Args:
            lhs: First selector
            rhs: Second selector
        """
        self.lhs = lhs
        self.rhs = rhs
        
    def is_member(self, id: int) -> bool:
        """Return true if both selectors match."""
        return self.lhs.is_member(id) and self.rhs.is_member(id)

class IDSelectorOr(IDSelector):
    """Combine two selectors with OR."""
    
    def __init__(self, lhs: IDSelector, rhs: IDSelector):
        """Initialize with two selectors.
        
        Args:
            lhs: First selector
            rhs: Second selector
        """
        self.lhs = lhs
        self.rhs = rhs
        
    def is_member(self, id: int) -> bool:
        """Return true if either selector matches."""
        return self.lhs.is_member(id) or self.rhs.is_member(id)

class IDSelectorXor(IDSelector):
    """Combine two selectors with XOR."""
    
    def __init__(self, lhs: IDSelector, rhs: IDSelector):
        """Initialize with two selectors.
        
        Args:
            lhs: First selector
            rhs: Second selector
        """
        self.lhs = lhs
        self.rhs = rhs
        
    def is_member(self, id: int) -> bool:
        """Return true if exactly one selector matches."""
        return self.lhs.is_member(id) != self.rhs.is_member(id)  # XOR
