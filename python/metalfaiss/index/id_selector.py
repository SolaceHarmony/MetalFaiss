"""
id_selector.py - ID selection and filtering

This implements the IDSelector functionality from FAISS, used to define subsets
of vectors to handle during search or removal operations.

Original: faiss/impl/IDSelector.h
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import mlx.core as mx

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
        ids: Sequence[int]
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

        # Use Python's bisect on the provided sequence
        sl = ids[:list_size]
        jmin = bisect_left(sl, self.imin)
        jmax = bisect_right(sl, self.imax - 1)
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

        # Initialize Bloom filter (bytes), then normalize to MLX
        # Use 8 bits per ID for good false positive rate
        self.nbits = max(64, 8 * len(ids))
        self.mask = (1 << self.nbits) - 1
        _bloom = bytearray(self.nbits // 8)

        # Add IDs to Bloom filter (Python control plane)
        for id in ids:
            h = id & self.mask  # Simple hash function
            _bloom[h >> 3] |= 1 << (h & 7)
        # Store as MLX array for GPU-friendly checks
        self.bloom = mx.array(list(_bloom), dtype=mx.uint8)
            
    def is_member(self, id: int) -> bool:
        """Check if ID is selected using Bloom filter."""
        # Check Bloom filter first
        h = id & self.mask
        byte = self.bloom[h >> 3]
        mask = mx.left_shift(mx.array(1, dtype=mx.uint8), mx.array(h & 7, dtype=mx.uint8))
        if not bool(mx.not_equal(mx.bitwise_and(byte, mask), mx.array(0, dtype=mx.uint8)).item()):  # boundary-ok
            return False  # Definitely not in set

        # Possible match, check actual set
        return id in self.id_set

class IDSelectorBitmap(IDSelector):
    """Select IDs using bitmap."""

    def __init__(self, n: int, bitmap: Union[Sequence[int], bytes, bytearray, mx.array]):
        """Initialize with bitmap.

        Args:
            n: Number of IDs
            bitmap: Bytes-like bitmap of ceil(n/8) bytes, sequence of ints, or `mx.array` of dtype uint8
        """
        self.n = n
        # Normalize to MLX array (uint8) for GPU-friendly bitwise ops
        if isinstance(bitmap, mx.array):
            self.bitmap = bitmap.astype(mx.uint8)
        elif isinstance(bitmap, (bytes, bytearray)):
            self.bitmap = mx.array(list(bitmap), dtype=mx.uint8)
        else:
            self.bitmap = mx.array([int(b) & 0xFF for b in bitmap], dtype=mx.uint8)

    def is_member(self, id: int) -> bool:
        """Check if ID is selected in bitmap."""
        if not (0 <= id < self.n):
            return False
        byte_idx = id >> 3  # Divide by 8
        bit_idx = id & 7    # Modulo 8
        byte = self.bitmap[byte_idx]
        mask = mx.left_shift(mx.array(1, dtype=mx.uint8), mx.array(bit_idx, dtype=mx.uint8))
        return bool(mx.not_equal(mx.bitwise_and(byte, mask), mx.array(0, dtype=mx.uint8)).item())  # boundary-ok

    def is_member_batch(self, ids: mx.array) -> mx.array:
        """Vectorized membership check for a batch of IDs.

        Args:
            ids: MLX array of integer IDs

        Returns:
            MLX boolean array indicating membership for each ID
        """
        ids = ids.astype(mx.int32)
        valid = mx.logical_and(mx.greater_equal(ids, mx.array(0, dtype=mx.int32)), mx.less(ids, mx.array(self.n, dtype=mx.int32)))
        byte_idx = mx.right_shift(ids, mx.array(3, dtype=mx.int32)).astype(mx.int32)
        bit_idx = mx.bitwise_and(ids, mx.array(7, dtype=mx.int32)).astype(mx.uint8)
        bytes_ = mx.take(self.bitmap, byte_idx)
        mask = mx.left_shift(mx.array(1, dtype=mx.uint8), bit_idx)
        bits = mx.not_equal(mx.bitwise_and(bytes_, mask), mx.array(0, dtype=mx.uint8))
        return mx.logical_and(valid, bits)

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
