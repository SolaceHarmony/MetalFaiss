"""
any_index.py - Generic index type that can be converted to specific types
"""

from typing import List, Optional, Union

from utils.search_result import SearchResult
from .base_index import BaseIndex
from .id_map import IDMap
from .id_map2 import IDMap2
from ..types.metric_type import MetricType

class AnyIndex(BaseIndex):
    """Generic index that can be converted to specific index types."""
    
    def __init__(self, index_pointer):
        self.index_pointer = index_pointer
        
    @property
    def d(self) -> int:
        return self.index_pointer.d
        
    def train(self, xs: List[List[float]]) -> None:
        self.index_pointer.train(xs)
        
    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        if ids is None:
            self.index_pointer.add(xs)
        else:
            self.index_pointer.add(xs, ids)
            
    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        return self.index_pointer.search(xs, k)
        
    def reset(self) -> None:
        self.index_pointer.reset()
        
    def reconstruct(self, key: int) -> List[float]:
        return self.index_pointer.reconstruct(key)
        
    def save_to_file(self, filename: str) -> None:
        self.index_pointer.save_to_file(filename)
        
    def clone(self) -> 'AnyIndex':
        return AnyIndex(self.index_pointer.clone())
