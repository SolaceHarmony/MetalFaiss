# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

from enum import Enum
from typing import Optional
import mlx.core as mx
from .index_error import IndexError
from .index import BaseIndex

class IOFlag(Enum):
    """Flags for controlling index I/O operations"""
    MMAP = 1
    READ_ONLY = 2

def load_index(
    filename: str, 
    io_flag: IOFlag = IOFlag.READ_ONLY
) -> 'BaseIndex':
    """
    Load an index from a file.
    
    Args:
        filename: Path to index file
        io_flag: I/O flag controlling how file is loaded
        
    Returns:
        Loaded index instance
        
    Raises:
        IndexError: Always - index I/O not implemented
        FileNotFoundError: If file doesn't exist
    """
    raise IndexError(
        "Index loading from file is not implemented. "
        "MetalFaiss focuses on in-memory operations."
    )
        
def save_index(index: 'BaseIndex', filename: str) -> None:
    """
    Save an index to a file.
    
    Args:
        index: Index to save
        filename: Path where index should be saved
        
    Raises:
        IndexError: Always - index I/O not implemented
    """
    raise IndexError(
        "Index saving to file is not implemented. "
        "MetalFaiss focuses on in-memory operations."
    )
