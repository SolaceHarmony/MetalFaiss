"""
errors.py - Error handling for MetalFaiss

This implements error handling that matches FAISS's behavior, with specific
error types and error codes for different failure modes.

Original: faiss/impl/FaissException.h, SwiftFaiss/IndexError.swift
"""

from enum import IntEnum
from typing import Optional

class ErrorCode(IntEnum):
    """Error codes matching FAISS."""
    SUCCESS = 0
    READONLY = 1          # Operation not allowed in read-only mode
    MEMORY = 2           # Memory allocation failed
    IO = 3              # I/O error
    RUNTIME = 4         # Runtime error
    NOT_IMPLEMENTED = 5  # Feature not implemented
    INVALID_ARGS = 6    # Invalid arguments
    BAD_STATE = 7       # Index in invalid state for operation
    ASSERTION = 8       # Internal assertion failed
    METRIC = 9          # Invalid metric type
    INDEX = 10          # Invalid index type
    THREAD = 11         # Threading error
    FILE_NOT_FOUND = 12 # File not found
    PERMISSION = 13     # Permission denied
    FORMAT = 14         # Invalid file format

class MetalFaissError(Exception):
    """Base class for MetalFaiss errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.RUNTIME
    ):
        """Initialize error.
        
        Args:
            message: Error message
            code: Error code
        """
        super().__init__(message)
        self.code = code
        self.message = message
        
    def __str__(self) -> str:
        return f"[Error {self.code.name}] {self.message}"

class ReadOnlyError(MetalFaissError):
    """Error raised when attempting to modify read-only index."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.READONLY)

class MemoryError(MetalFaissError):
    """Error raised when memory allocation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.MEMORY)

class IOError(MetalFaissError):
    """Error raised for I/O operations."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.IO)

class NotImplementedError(MetalFaissError):
    """Error raised for unimplemented features."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.NOT_IMPLEMENTED)

class InvalidArgumentError(MetalFaissError):
    """Error raised for invalid arguments."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.INVALID_ARGS)

class BadStateError(MetalFaissError):
    """Error raised when index is in invalid state."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.BAD_STATE)

class AssertionError(MetalFaissError):
    """Error raised for internal assertions."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.ASSERTION)

class MetricError(MetalFaissError):
    """Error raised for invalid metric types."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.METRIC)

class IndexError(MetalFaissError):
    """Error raised for invalid index operations."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.INDEX)

class ThreadError(MetalFaissError):
    """Error raised for threading issues."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.THREAD)

class FileNotFoundError(MetalFaissError):
    """Error raised when file not found."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.FILE_NOT_FOUND)

class PermissionError(MetalFaissError):
    """Error raised for permission issues."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.PERMISSION)

class FormatError(MetalFaissError):
    """Error raised for invalid file formats."""
    
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.FORMAT)

# Global error state
_last_error: Optional[MetalFaissError] = None

def set_last_error(error: MetalFaissError) -> None:
    """Set last error for error checking."""
    global _last_error
    _last_error = error

def get_last_error() -> Optional[MetalFaissError]:
    """Get last error that occurred."""
    return _last_error

def clear_last_error() -> None:
    """Clear last error state."""
    global _last_error
    _last_error = None

def check_error(code: ErrorCode) -> None:
    """Check error code and raise appropriate exception.
    
    Args:
        code: Error code to check
        
    Raises:
        MetalFaissError: If error code indicates failure
    """
    if code != ErrorCode.SUCCESS:
        error = get_last_error()
        if error is None:
            error = MetalFaissError(
                f"Unknown error occurred (code {code})",
                code
            )
        clear_last_error()
        raise error