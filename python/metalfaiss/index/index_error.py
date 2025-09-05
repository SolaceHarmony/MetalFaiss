"""
index_error.py - Error classes for MetalFaiss indices
"""

class IndexError(Exception):
    """Base class for index errors."""
    pass

class TrainingError(IndexError):
    """Error raised when index training fails."""
    pass

class MemoryError(IndexError):
    """Error raised when index runs out of memory."""
    pass

class DimensionError(IndexError):
    """Error raised when vector dimensions don't match."""
    pass

class NotImplementedError(IndexError):
    """Error raised when operation is not implemented."""
    pass

class InvalidArgumentError(IndexError):
    """Error raised when invalid argument is provided."""
    pass

class RuntimeError(IndexError):
    """Error raised when operation fails at runtime."""
    pass

class IOError(IndexError):
    """Error raised when I/O operation fails."""
    pass

class StateError(IndexError):
    """Error raised when index is in invalid state."""
    pass

class NotTrainedError(StateError):
    """Error raised when untrained index is used."""
    pass

class EmptyIndexError(StateError):
    """Error raised when empty index is used."""
    pass

class ReadOnlyError(StateError):
    """Error raised when modifying read-only index."""
    pass

class CapacityError(StateError):
    """Error raised when index capacity is exceeded."""
    pass

class MetricTypeError(InvalidArgumentError):
    """Error raised when invalid metric type is used."""
    pass

class QuantizerTypeError(InvalidArgumentError):
    """Error raised when invalid quantizer type is used."""
    pass

class RangeStatError(InvalidArgumentError):
    """Error raised when invalid range stat is used."""
    pass

class DimensionMismatchError(DimensionError):
    """Error raised when vector dimensions don't match index."""
    pass

class SubDimensionError(DimensionError):
    """Error raised when sub-vector dimensions are invalid."""
    pass

class GPUError(RuntimeError):
    """Error raised when GPU operation fails."""
    pass

class GPUMemoryError(MemoryError):
    """Error raised when GPU runs out of memory."""
    pass

class GPUNotSupportedError(NotImplementedError):
    """Error raised when GPU operation is not supported."""
    pass
