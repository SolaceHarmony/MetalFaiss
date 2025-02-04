"""
heap.py

This module provides heap utilities for MetalFaiss, a Python port of FAISS
using MLX. These functions implement min-heap and max-heap operations on MLX arrays,
mirroring the functionality found in faiss/utils/Heap.h.

All arrays here are assumed to be 1D MLX arrays (mx.array) of appropriate length.
"""

import mlx.core as mx
from typing import Tuple

def minheap_push(k: int, bh_val: mx.array, bh_ids: mx.array, val: float, idx: int) -> None:
    """
    Push a new element (val, idx) into the min-heap represented by bh_val and bh_ids.
    
    The heap is stored in 0-indexed arrays of length k.
    This operation assumes that the heap already contains k-1 elements,
    and we insert the new element by “bubbling up.”
    """
    i = k - 1
    while i > 0:
        parent = (i - 1) // 2
        if float(bh_val[parent]) <= val:
            break
        bh_val[i] = bh_val[parent]
        bh_ids[i] = bh_ids[parent]
        i = parent
    bh_val[i] = val   # Direct scalar assignment
    bh_ids[i] = idx

def minheap_pop(k: int, bh_val: mx.array, bh_ids: mx.array) -> Tuple[float, int]:
    """
    Pop and return the smallest element from the min-heap.
    After removal, the heap property is maintained.
    
    Returns:
        A tuple (min_val, min_idx)
    """
    min_val = float(bh_val[0])
    min_idx = int(bh_ids[0])
    last_val = float(bh_val[k - 1])
    last_idx = int(bh_ids[k - 1])
    i = 0
    j = 1  # left child index
    while j < k - 1:
        if j + 1 < k - 1 and float(bh_val[j + 1]) < float(bh_val[j]):
            j = j + 1
        if last_val <= float(bh_val[j]):
            break
        bh_val[i] = bh_val[j]
        bh_ids[i] = bh_ids[j]
        i = j
        j = 2 * i + 1
    bh_val[i] = last_val
    bh_ids[i] = last_idx
    return min_val, min_idx

def heap_heapify(k: int, bh_val: mx.array, bh_ids: mx.array) -> None:
    """
    Given arrays bh_val and bh_ids of length k, convert them in-place into a valid min-heap.
    Uses standard bottom-up heap construction.
    """
    for i in reversed(range(k // 2)):
        _bubble_down(i, k, bh_val, bh_ids)

def _bubble_down(i: int, k: int, bh_val: mx.array, bh_ids: mx.array) -> None:
    """
    Helper function: bubble down the element at index i to restore the heap property.
    """
    val = float(bh_val[i])
    idx = int(bh_ids[i])
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < k and float(bh_val[left]) < float(bh_val[smallest]):
            smallest = left
        if right < k and float(bh_val[right]) < float(bh_val[smallest]):
            smallest = right
        if smallest == i:
            break
        bh_val[i] = bh_val[smallest]
        bh_ids[i] = bh_ids[smallest]
        i = smallest
    bh_val[i] = val
    bh_ids[i] = idx

def heap_replace_top(k: int, bh_val: mx.array, bh_ids: mx.array, val: float, idx: int) -> None:
    """
    Replace the top element (minimum) of the heap with (val, idx)
    and restore the heap property by bubbling down.
    """
    bh_val[0] = val
    bh_ids[0] = idx
    _bubble_down(0, k, bh_val, bh_ids)

def heap_reorder(k: int, bh_val: mx.array, bh_ids: mx.array) -> None:
    """
    Reorder the heap arrays so that the elements are in sorted order.
    This extracts the heap elements one-by-one and stores them in increasing order.
    """
    vals = [float(bh_val[i]) for i in range(k)]
    indices = [int(bh_ids[i]) for i in range(k)]
    sorted_pairs = sorted(zip(vals, indices), key=lambda pair: pair[0])
    for i, (val, idx) in enumerate(sorted_pairs):
        bh_val[i] = val
        bh_ids[i] = idx

class HeapArray:
    """
    A container for a set of heaps.
    
    Each heap is represented by two MLX arrays (one for values, one for indices).
    The heaps are stored contiguously in two 1D MLX arrays.
    """
    def __init__(self, nh: int, k: int, is_min_heap: bool = True):
        """
        Initialize the HeapArray.
        
        Args:
            nh: Number of heaps (e.g., one per query)
            k: Number of elements per heap (the “capacity”)
            is_min_heap: Whether this is a min-heap (True) or max-heap (False)
        """
        self.nh = nh
        self.k = k
        self.is_min_heap = is_min_heap
        neutral = float("inf") if is_min_heap else -float("inf")
        # Use MLX's dtype constants, not NumPy dtypes.
        self.val = mx.array([neutral] * (nh * k), dtype=mx.float32)
        self.ids = mx.array([-1] * (nh * k), dtype=mx.int32)
    
    def get_heap(self, i: int) -> Tuple[mx.array, mx.array]:
        """
        Get the value and index arrays for heap number i.
        """
        start = i * self.k
        end = start + self.k
        return self.val[start:end], self.ids[start:end]
    
    def heapify_all(self) -> None:
        """
        Heapify all heaps.
        """
        for i in range(self.nh):
            heap_heapify(self.k, *self.get_heap(i))
    
    def replace_top(self, i: int, val: float, idx: int) -> None:
        """
        Replace the top element of heap i.
        """
        heap_replace_top(self.k, *self.get_heap(i), val, idx)
    
    def reorder_all(self) -> None:
        """
        Reorder all heaps so that each heap is in sorted order.
        """
        for i in range(self.nh):
            heap_reorder(self.k, *self.get_heap(i))


if __name__ == "__main__":
    # Simple unit test for the heap functions.
    k = 8
    bh_val = mx.array([float("inf")] * k, dtype=mx.float32)
    bh_ids = mx.array([-1] * k, dtype=mx.int32)
    values = [5.0, 3.2, 7.8, 1.5, 2.9, 4.4, 6.1, 0.8]
    for i, v in enumerate(values):
        minheap_push(i + 1, bh_val, bh_ids, v, i)
    print("Heap values after push:")
    print([float(bh_val[i]) for i in range(k)])
    print("Heap indices:")
    print([int(bh_ids[i]) for i in range(k)])
    
    min_val, min_idx = minheap_pop(k, bh_val, bh_ids)
    print("Popped element:", min_val, min_idx)
    
    heap_reorder(k - 1, bh_val, bh_ids)
    print("Sorted heap values:")
    print([float(bh_val[i]) for i in range(k - 1)])
    print("Sorted heap indices:")
    print([int(bh_ids[i]) for i in range(k - 1)])