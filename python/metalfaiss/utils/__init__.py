"""
utils - Utility functions and classes for MetalFaiss

This package provides various utilities:
- Search result handling
- Binary I/O
- Memory management
- Error handling
"""

from .search_result import (
    SearchResult,
    SearchRangeResult
)

from .binary_io import (
    write_binary,
    read_binary,
    write_binary_vector,
    read_binary_vector
)

__all__ = [
    'SearchResult',
    'SearchRangeResult',
    'write_binary',
    'read_binary',
    'write_binary_vector',
    'read_binary_vector'
]