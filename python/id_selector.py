import mlx.core as mx
import numpy as np

class IDSelector:
    """
    Class for selecting vectors by their IDs using different selection methods.
    """
    
    def __init__(self):
        self.selected_ids = None
        
    @classmethod
    def range(cls, start: int, end: int) -> 'IDSelector':
        """
        Create an IDSelector that selects IDs within a range.
        
        Args:
            start: Lower bound (inclusive)
            end: Upper bound (exclusive)
            
        Returns:
            IDSelector instance for range selection
        """
        selector = cls()
        selector.selected_ids = mx.arange(start, end, dtype=mx.int64)
        return selector
        
    @classmethod
    def batch(cls, ids: list[int]) -> 'IDSelector':
        """
        Create an IDSelector that selects specific IDs.
        
        Args:
            ids: List of IDs to select
            
        Returns:
            IDSelector instance for batch selection
        """
        selector = cls()
        selector.selected_ids = mx.array(ids, dtype=mx.int64)
        return selector
        
    def is_selected(self, id: int) -> bool:
        """
        Check if an ID is selected by this selector.
        
        Args:
            id: ID to check
            
        Returns:
            True if ID is selected, False otherwise
        """
        if self.selected_ids is None:
            return False
        return id in self.selected_ids
        
    def select_vectors(self, vectors: mx.array, ids: mx.array) -> mx.array:
        """
        Select vectors based on their IDs.
        
        Args:
            vectors: Array of vectors to select from
            ids: Array of corresponding IDs
            
        Returns:
            Selected vectors
        """
        if self.selected_ids is None:
            return mx.array([], dtype=vectors.dtype)
        mask = mx.any(ids[:, None] == self.selected_ids[None, :], axis=1)
        return vectors[mask]
