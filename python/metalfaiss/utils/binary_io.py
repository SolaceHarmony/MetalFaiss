"""
binary_io.py - Binary I/O utilities for MetalFaiss
"""

import numpy as np  # boundary-ok (I/O conversion only)
import struct
from typing import BinaryIO, Tuple, Union
import mlx.core as mx

def write_binary(f: BinaryIO, data: Union[bytes, bytearray, memoryview]) -> None:
    """Write binary data to file.
    
    Args:
        f: File object to write to
        data: Binary data to write
    """
    f.write(data)

def read_binary(f: BinaryIO, size: int) -> bytes:
    """Read binary data from file.
    
    Args:
        f: File object to read from
        size: Number of bytes to read
        
    Returns:
        Binary data read from file
    """
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data

def write_binary_vector(f: BinaryIO, x: mx.array) -> None:
    """Write binary vector to file.
    
    Args:
        f: File object to write to
        x: Vector to write
    """
    # Write shape
    shape = x.shape
    f.write(struct.pack('Q' * len(shape), *shape))
    
    # Write dtype
    dtype_str = str(x.dtype).encode('ascii')
    f.write(struct.pack('B', len(dtype_str)))
    f.write(dtype_str)
    
    # Write data
    write_binary(f, x.numpy().tobytes())  # boundary-ok

def read_binary_vector(f: BinaryIO) -> mx.array:
    """Read binary vector from file.
    
    Args:
        f: File object to read from
        
    Returns:
        Vector read from file
    """
    # Read shape
    ndim = struct.unpack('Q', read_binary(f, 8))[0]
    shape = struct.unpack('Q' * ndim, read_binary(f, 8 * ndim))
    
    # Read dtype
    dtype_len = struct.unpack('B', read_binary(f, 1))[0]
    dtype_str = read_binary(f, dtype_len).decode('ascii')
    
    # Read data
    size = np.prod(shape) * np.dtype(dtype_str).itemsize
    data = read_binary(f, size)
    
    # Convert to array
    x = np.frombuffer(data, dtype=dtype_str).reshape(shape)
    return mx.array(x)

def write_binary_vectors(f: BinaryIO, xs: mx.array) -> None:
    """Write multiple binary vectors to file.
    
    Args:
        f: File object to write to
        xs: Vectors to write (n, d)
    """
    # Write number of vectors
    f.write(struct.pack('Q', len(xs)))
    
    # Write each vector
    for x in xs:
        write_binary_vector(f, x)

def read_binary_vectors(f: BinaryIO) -> mx.array:
    """Read multiple binary vectors from file.
    
    Args:
        f: File object to read from
        
    Returns:
        Vectors read from file (n, d)
    """
    # Read number of vectors
    n = struct.unpack('Q', read_binary(f, 8))[0]
    
    # Read each vector
    xs = []
    for _ in range(n):
        xs.append(read_binary_vector(f))
        
    # Stack vectors
    return mx.stack(xs)
