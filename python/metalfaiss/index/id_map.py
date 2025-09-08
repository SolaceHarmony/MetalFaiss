"""
id_map.py - ID mapping functionality

This implements ID mapping from FAISS, allowing translation between internal
and external IDs. IDMap provides basic one-way mapping, while IDMap2 adds
efficient reconstruction via two-way mapping.

Original: faiss/IndexIDMap.h
"""

from typing import List, Dict, Optional, Any
import mlx.core as mx
from .base_index import BaseIndex
from ..utils.search_result import SearchResult, SearchRangeResult
from ..types.metric_type import MetricType
from ..errors import InvalidArgumentError

class IDMap(BaseIndex):
    """Index wrapper that translates internal to external IDs.
    
    This provides one-way mapping from internal to external IDs, allowing
    the wrapped index to use sequential IDs internally while presenting
    arbitrary IDs to the user.
    """
    
    def __init__(self, index: BaseIndex):
        """Initialize ID map wrapper.
        
        Args:
            index: Base index to wrap
        """
        super().__init__(index.d)
        self.index = index
        self.id_map: List[int] = []  # External IDs
        self.own_fields = True
        
    def add(self, xs: List[List[float]]) -> None:
        """Adding vectors without IDs is not supported.
        
        Raises:
            InvalidArgumentError: Always, use add_with_ids instead
        """
        raise InvalidArgumentError(
            "IDMap requires explicit IDs. Use add_with_ids instead."
        )
        
    def add_with_ids(
        self,
        xs: List[List[float]],
        ids: List[int]
    ) -> None:
        """Add vectors with external IDs.
        
        Args:
            xs: Vectors to add
            ids: External IDs for vectors
            
        Raises:
            ValueError: If length of xs and ids don't match
        """
        if len(xs) != len(ids):
            raise ValueError("Number of vectors and IDs must match")
            
        # Add vectors to base index
        self.index.add(xs)
        
        # Store external IDs
        self.id_map.extend(ids)
        self._ntotal = len(self.id_map)
        
    def search(
        self,
        xs: List[List[float]],
        k: int
    ) -> SearchResult:
        """Search and translate internal IDs to external IDs.
        
        Args:
            xs: Query vectors
            k: Number of neighbors
            
        Returns:
            SearchResult with external IDs
        """
        # Search in base index
        result = self.index.search(xs, k)
        
        # Translate internal IDs to external IDs
        translated_labels = [
            [self.id_map[idx] if idx >= 0 else -1 for idx in query_labels]
            for query_labels in result.labels
        ]
        
        return SearchResult(
            distances=result.distances,
            labels=translated_labels
        )
        
    def range_search(
        self,
        xs: List[List[float]],
        radius: float
    ) -> SearchRangeResult:
        """Range search with ID translation.
        
        Args:
            xs: Query vectors
            radius: Search radius
            
        Returns:
            SearchRangeResult with external IDs
        """
        # Range search in base index
        result = self.index.range_search(xs, radius)
        
        # Translate internal IDs to external IDs
        translated_labels = [
            [self.id_map[idx] if idx >= 0 else -1 for idx in query_labels]
            for query_labels in result.labels
        ]
        
        return SearchRangeResult(
            lims=result.lims,
            distances=result.distances,
            labels=translated_labels
        )
        
    def remove_ids(self, sel: Any) -> int:
        """Remove vectors selected by ID selector.
        
        Args:
            sel: ID selector
            
        Returns:
            Number of vectors removed
            
        Raises:
            NotImplementedError: If base index doesn't support removal
        """
        # Create translated selector that works with internal IDs
        translated_sel = IDSelectorTranslated(self.id_map, sel)
        
        # Remove from base index
        n_removed = self.index.remove_ids(translated_sel)
        
        # Update ID map
        if n_removed > 0:
            new_map = []
            for i, id in enumerate(self.id_map):
                if not sel.is_member(id):
                    new_map.append(id)
            self.id_map = new_map
            self._ntotal = len(self.id_map)
            
        return n_removed
        
    def train(self, xs: List[List[float]]) -> None:
        """Train base index.
        
        Args:
            xs: Training vectors
        """
        self.index.train(xs)
        
    def reset(self) -> None:
        """Reset index and ID map."""
        self.index.reset()
        self.id_map = []
        self._ntotal = 0

class IDMap2(IDMap):
    """Index wrapper with two-way ID mapping.
    
    This extends IDMap with a reverse mapping from external to internal IDs,
    allowing efficient reconstruction of vectors by external ID.
    """
    
    def __init__(self, index: BaseIndex):
        """Initialize two-way ID map wrapper.
        
        Args:
            index: Base index to wrap
        """
        super().__init__(index)
        self.rev_map: Dict[int, int] = {}  # External ID -> internal ID
        
    def add_with_ids(
        self,
        xs: List[List[float]],
        ids: List[int]
    ) -> None:
        """Add vectors with two-way ID mapping.
        
        Args:
            xs: Vectors to add
            ids: External IDs for vectors
            
        Raises:
            ValueError: If length of xs and ids don't match or IDs not unique
        """
        if len(xs) != len(ids):
            raise ValueError("Number of vectors and IDs must match")
            
        # Check ID uniqueness
        if len(set(ids)) != len(ids):
            raise ValueError("IDs must be unique")
            
        # Add vectors to base index
        start_idx = self._ntotal
        self.index.add(xs)
        
        # Update both mappings
        for i, id in enumerate(ids):
            self.id_map.append(id)
            self.rev_map[id] = start_idx + i
            
        self._ntotal = len(self.id_map)
        
    def reconstruct(self, key: int) -> List[float]:
        """Reconstruct vector by external ID.
        
        Args:
            key: External ID
            
        Returns:
            Reconstructed vector
            
        Raises:
            KeyError: If ID not found
        """
        if key not in self.rev_map:
            raise KeyError(f"ID {key} not found")
            
        return self.index.reconstruct(self.rev_map[key])
        
    def remove_ids(self, sel: Any) -> int:
        """Remove vectors with two-way mapping update.
        
        Args:
            sel: ID selector
            
        Returns:
            Number of vectors removed
        """
        n_removed = super().remove_ids(sel)
        
        # Rebuild reverse map
        if n_removed > 0:
            self.rev_map.clear()
            for i, id in enumerate(self.id_map):
                self.rev_map[id] = i
                
        return n_removed
        
    def reset(self) -> None:
        """Reset both mappings."""
        super().reset()
        self.rev_map.clear()
        
    def check_consistency(self) -> None:
        """Check consistency of forward and reverse mappings.
        
        Raises:
            RuntimeError: If mappings are inconsistent
        """
        if len(self.id_map) != len(self.rev_map):
            raise RuntimeError("Forward and reverse maps have different sizes")
            
        for i, id in enumerate(self.id_map):
            if id not in self.rev_map:
                raise RuntimeError(f"ID {id} missing from reverse map")
            if self.rev_map[id] != i:
                raise RuntimeError(
                    f"Inconsistent mapping for ID {id}: "
                    f"forward={i}, reverse={self.rev_map[id]}"
                )

class IDSelectorTranslated:
    """ID selector that works with translated IDs.
    
    This wraps an ID selector that works with external IDs to work with
    internal IDs used by the base index.
    """
    
    def __init__(self, id_map: List[int], sel: Any):
        """Initialize translated selector.
        
        Args:
            id_map: ID mapping from internal to external IDs
            sel: Selector that works with external IDs
        """
        self.id_map = id_map
        self.sel = sel
        
    def is_member(self, id: int) -> bool:
        """Check if internal ID is selected.
        
        Args:
            id: Internal ID
            
        Returns:
            True if corresponding external ID is selected
        """
        return self.sel.is_member(self.id_map[id])
