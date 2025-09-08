"""
index_io.py - Index I/O functions for MetalFaiss
"""

import os
import json
import struct
import mlx.core as mx
from enum import Enum, auto
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

from .flat_index import FlatIndex
from .ivf_flat_index import IVFFlatIndex
from .binary_flat_index import BinaryFlatIndex
from .binary_ivf_index import BinaryIVFIndex
from .product_quantizer_index import ProductQuantizerIndex
from .scalar_quantizer_index import ScalarQuantizerIndex
from ..vector_transform import BaseVectorTransform
from ..types.metric_type import MetricType
from ..types.quantizer_type import QuantizerType
from ..utils.binary_io import write_binary, read_binary

class IOFlag(Enum):
    """I/O flags for index serialization."""
    IO_FLAG_NONE = auto()
    IO_FLAG_MMAP = auto()
    IO_FLAG_ONDISK = auto()
    IO_FLAG_MEM_RESIDENT = auto()
    IO_FLAG_MMAP_RESIDENT = auto()

def write_index(index: 'BaseIndex', fname: str) -> None:
    """Write index to file.
    
    Args:
        index: Index to write
        fname: File name to write to
    """
    with open(fname, 'wb') as f:
        # Write header
        header = {
            'type': type(index).__name__,
            'd': index.d,
            'ntotal': index.ntotal,
            'is_trained': index.is_trained
        }
        
        # Add metric type if available
        if hasattr(index, 'metric'):
            header['metric'] = index.metric.value
            
        # Add quantizer type if available
        if hasattr(index, 'qtype'):
            header['qtype'] = index.qtype.value
            
        # Write header
        header_bytes = json.dumps(header).encode('utf-8')
        write_binary(f, struct.pack('Q', len(header_bytes)))
        write_binary(f, header_bytes)
        
        # Write index-specific data
        index.write(f)

def read_index(fname: str) -> 'BaseIndex':
    """Read index from file.
    
    Args:
        fname: File name to read from
        
    Returns:
        Loaded index
    """
    with open(fname, 'rb') as f:
        # Read header
        header_size = struct.unpack('Q', read_binary(f, 8))[0]
        header = json.loads(read_binary(f, header_size).decode('utf-8'))
        
        # Create index
        index_type = header['type']
        d = header['d']
        
        # Get metric type if available
        metric = None
        if 'metric' in header:
            metric = MetricType(header['metric'])
            
        # Get quantizer type if available
        qtype = None
        if 'qtype' in header:
            qtype = QuantizerType(header['qtype'])
            
        # Create index based on type
        if index_type == 'FlatIndex':
            index = FlatIndex(d, metric=metric)
        elif index_type == 'IVFFlatIndex':
            index = IVFFlatIndex(d, metric=metric)
        elif index_type == 'BinaryFlatIndex':
            index = BinaryFlatIndex(d)
        elif index_type == 'BinaryIVFIndex':
            index = BinaryIVFIndex(d)
        elif index_type == 'ProductQuantizerIndex':
            index = ProductQuantizerIndex(d, metric=metric)
        elif index_type == 'ScalarQuantizerIndex':
            index = ScalarQuantizerIndex(d, qtype=qtype, metric=metric)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        # Read index-specific data
        index.read(f)
        
        # Verify header info
        assert index.ntotal == header['ntotal']
        assert index.is_trained == header['is_trained']
        
        return index

__all__ = ['write_index', 'read_index', 'IOFlag']
