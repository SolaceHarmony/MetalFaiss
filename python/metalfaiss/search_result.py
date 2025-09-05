# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

from typing import List

class SearchResult:
    """Container for search results including distances and labels.
    Matches Swift implementation with list-of-lists structure."""
    
    def __init__(self, distances: List[List[float]], labels: List[List[int]]):
        """Initialize search result.
        
        Args:
            distances: Distance values for matches [n_queries][k]
            labels: Label/ID values for matches [n_queries][k]
        """
        self.distances = distances
        self.labels = labels
    
    def __eq__(self, other: 'SearchResult') -> bool:
        """Compare two search results for equality."""
        if not isinstance(other, SearchResult):
            return False
        return (self.distances == other.distances and 
                self.labels == other.labels)
    
    def __hash__(self) -> int:
        """Hash function for SearchResult."""
        return hash((str(self.distances), str(self.labels)))
