# heap.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license (see LICENSE file in the root directory).

import mlx.core as mx
import numpy as np
from typing import Tuple

class HeapArray:
    """
    A simple heap array structure for collecting search results.
    This class maintains two MLX arrays:
      - val: the heap values (e.g. distances or scores)
      - ids: corresponding indices (e.g. vector IDs)
    The heaps are stored as 2D arrays (nh x k) for nh queries each with k results.
    
    The operations here are implemented using MLX functions and are not fully optimized
    for performance but serve as a faithful starting point.
    """
    def __init__(self, nh: int, k: int, is_min_heap: bool = True):
        self.nh = nh
        self.k = k
        self.is_min_heap = is_min_heap
        # Initialize the heap arrays.
        # For a min-heap (e.g. for L2 distances) we initialize with a large constant.
        # For a max-heap (e.g. for inner product search) we initialize with a small constant.
        init_val = 1e20 if is_min_heap else -1e20
        self.val = mx.full((nh, k), init_val, dtype=mx.float32)
        self.ids = mx.full((nh, k), -1, dtype=mx.int64)

    def heapify(self) -> None:
        """
        Reorder each row of the heap arrays so that the heap property holds.
        In our simple version, we simply sort each row.
        For a min-heap, the smallest value will appear in the first column.
        """
        # Get the sorted indices along the last axis.
        sorted_indices = mx.argsort(self.val, axis=1)
        self.val = mx.take(self.val, sorted_indices, axis=1)
        self.ids = mx.take(self.ids, sorted_indices, axis=1)
        # Force evaluation to ensure the new order is computed immediately.
        mx.eval(self.val)
        mx.eval(self.ids)

    def add_result(self, row: int, new_val: float, new_id: int) -> None:
        """
        Try to insert a new (value, id) pair into the heap for a given query row.
        For a min-heap, if the new value is greater than the current minimum (at index 0),
        we replace it and then re-heapify the row.
        For a max-heap, we do the analogous operation.
        
        Args:
            row: The row (query) index.
            new_val: The new score/distance value.
            new_id: The corresponding index.
        """
        current = self.val[row, 0]
        if self.is_min_heap:
            if new_val > current:
                # Replace the root and re-heapify.
                self.val[row, 0] = new_val
                self.ids[row, 0] = new_id
                self.heapify()
        else:
            if new_val < current:
                self.val[row, 0] = new_val
                self.ids[row, 0] = new_id
                self.heapify()

    def reorder(self) -> None:
        """
        Reorder the heap arrays so that the best results (depending on the heap type)
        are in order. This simply calls heapify.
        """
        self.heapify()

# Helper functions for min-heap and max-heap operations.
def minheap_push(heap: HeapArray, row: int, new_val: float, new_id: int) -> None:
    """
    Insert a new value in a min-heap.
    """
    if new_val > heap.val[row, 0]:
        heap.add_result(row, new_val, new_id)

def minheap_pop(heap: HeapArray, row: int) -> Tuple[float, int]:
    """
    Pop the smallest element from a min-heap (the first element).
    Returns:
        A tuple (value, index) of the popped element.
    """
    # For simplicity, extract the first column and then shift the rest.
    val = heap.val[row, 0].asnumpy()[0]
    idx = heap.ids[row, 0].asnumpy()[0]
    # Shift the heap: remove the first column and append a default value.
    new_val_row = mx.concatenate([heap.val[row, 1:], mx.full((1,), 1e20, dtype=mx.float32)], axis=0)
    new_ids_row = mx.concatenate([heap.ids[row, 1:], mx.full((1,), -1, dtype=mx.int64)], axis=0)
    heap.val[row] = new_val_row
    heap.ids[row] = new_ids_row
    heap.heapify()
    return val, idx

def maxheap_push(heap: HeapArray, row: int, new_val: float, new_id: int) -> None:
    """
    Insert a new value in a max-heap.
    """
    if new_val < heap.val[row, 0]:
        heap.add_result(row, new_val, new_id)

def maxheap_pop(heap: HeapArray, row: int) -> Tuple[float, int]:
    """
    Pop the largest element from a max-heap.
    """
    val = heap.val[row, 0].asnumpy()[0]
    idx = heap.ids[row, 0].asnumpy()[0]
    new_val_row = mx.concatenate([heap.val[row, 1:], mx.full((1,), -1e20, dtype=mx.float32)], axis=0)
    new_ids_row = mx.concatenate([heap.ids[row, 1:], mx.full((1,), -1, dtype=mx.int64)], axis=0)
    heap.val[row] = new_val_row
    heap.ids[row] = new_ids_row
    heap.heapify()
    return val, idx

# A simple test if run as script.
if __name__ == "__main__":
    # Create a min-heap for 2 queries with 4 results each.
    heap = HeapArray(nh=2, k=4, is_min_heap=True)
    
    # Insert some sample values.
    heap.add_result(0, 10.0, 101)
    heap.add_result(0, 20.0, 102)
    heap.add_result(0, 15.0, 103)
    heap.add_result(0, 25.0, 104)
    
    print("Heap values (query 0):", heap.val[0].asnumpy())
    print("Heap indices (query 0):", heap.ids[0].asnumpy())
    
    # Push a new value.
    minheap_push(heap, 0, 18.0, 105)
    print("After push, heap values (query 0):", heap.val[0].asnumpy())
    print("After push, heap indices (query 0):", heap.ids[0].asnumpy())
    
    # Pop the smallest element.
    popped_val, popped_id = minheap_pop(heap, 0)
    print("Popped value:", popped_val, "Popped index:", popped_id)
    print("After pop, heap values (query 0):", heap.val[0].asnumpy())
    print("After pop, heap indices (query 0):", heap.ids[0].asnumpy())