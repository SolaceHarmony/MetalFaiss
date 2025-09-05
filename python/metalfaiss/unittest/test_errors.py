"""
test_errors.py - Tests for error handling

These tests verify that our error handling matches FAISS's behavior,
particularly around:
- Error codes
- Error messages
- Error state management
- Exception hierarchy
"""

import unittest
from ..errors import (
    ErrorCode,
    MetalFaissError,
    ReadOnlyError,
    MemoryError,
    IOError,
    NotImplementedError,
    InvalidArgumentError,
    BadStateError,
    AssertionError,
    MetricError,
    IndexError,
    ThreadError,
    FileNotFoundError,
    PermissionError,
    FormatError,
    set_last_error,
    get_last_error,
    clear_last_error,
    check_error
)

class TestErrors(unittest.TestCase):
    """Test error handling functionality."""
    
    def setUp(self):
        """Clear error state before each test."""
        clear_last_error()
        
    def test_error_codes(self):
        """Test error code values match FAISS."""
        self.assertEqual(ErrorCode.SUCCESS, 0)
        self.assertEqual(ErrorCode.READONLY, 1)
        self.assertEqual(ErrorCode.MEMORY, 2)
        self.assertEqual(ErrorCode.IO, 3)
        self.assertEqual(ErrorCode.RUNTIME, 4)
        self.assertEqual(ErrorCode.NOT_IMPLEMENTED, 5)
        self.assertEqual(ErrorCode.INVALID_ARGS, 6)
        self.assertEqual(ErrorCode.BAD_STATE, 7)
        self.assertEqual(ErrorCode.ASSERTION, 8)
        self.assertEqual(ErrorCode.METRIC, 9)
        self.assertEqual(ErrorCode.INDEX, 10)
        self.assertEqual(ErrorCode.THREAD, 11)
        self.assertEqual(ErrorCode.FILE_NOT_FOUND, 12)
        self.assertEqual(ErrorCode.PERMISSION, 13)
        self.assertEqual(ErrorCode.FORMAT, 14)
        
    def test_error_messages(self):
        """Test error message formatting."""
        error = MetalFaissError("Test error")
        self.assertEqual(str(error), "[Error RUNTIME] Test error")
        
        error = ReadOnlyError("Cannot modify")
        self.assertEqual(str(error), "[Error READONLY] Cannot modify")
        
    def test_error_hierarchy(self):
        """Test error class hierarchy."""
        # All errors should inherit from MetalFaissError
        self.assertTrue(issubclass(ReadOnlyError, MetalFaissError))
        self.assertTrue(issubclass(MemoryError, MetalFaissError))
        self.assertTrue(issubclass(IOError, MetalFaissError))
        self.assertTrue(issubclass(NotImplementedError, MetalFaissError))
        self.assertTrue(issubclass(InvalidArgumentError, MetalFaissError))
        self.assertTrue(issubclass(BadStateError, MetalFaissError))
        self.assertTrue(issubclass(AssertionError, MetalFaissError))
        self.assertTrue(issubclass(MetricError, MetalFaissError))
        self.assertTrue(issubclass(IndexError, MetalFaissError))
        self.assertTrue(issubclass(ThreadError, MetalFaissError))
        self.assertTrue(issubclass(FileNotFoundError, MetalFaissError))
        self.assertTrue(issubclass(PermissionError, MetalFaissError))
        self.assertTrue(issubclass(FormatError, MetalFaissError))
        
        # Each error should have correct code
        self.assertEqual(ReadOnlyError("test").code, ErrorCode.READONLY)
        self.assertEqual(MemoryError("test").code, ErrorCode.MEMORY)
        self.assertEqual(IOError("test").code, ErrorCode.IO)
        self.assertEqual(NotImplementedError("test").code, ErrorCode.NOT_IMPLEMENTED)
        self.assertEqual(InvalidArgumentError("test").code, ErrorCode.INVALID_ARGS)
        self.assertEqual(BadStateError("test").code, ErrorCode.BAD_STATE)
        self.assertEqual(AssertionError("test").code, ErrorCode.ASSERTION)
        self.assertEqual(MetricError("test").code, ErrorCode.METRIC)
        self.assertEqual(IndexError("test").code, ErrorCode.INDEX)
        self.assertEqual(ThreadError("test").code, ErrorCode.THREAD)
        self.assertEqual(FileNotFoundError("test").code, ErrorCode.FILE_NOT_FOUND)
        self.assertEqual(PermissionError("test").code, ErrorCode.PERMISSION)
        self.assertEqual(FormatError("test").code, ErrorCode.FORMAT)
        
    def test_error_state(self):
        """Test error state management."""
        # Initially no error
        self.assertIsNone(get_last_error())
        
        # Set error
        error = MetalFaissError("Test error")
        set_last_error(error)
        self.assertEqual(get_last_error(), error)
        
        # Clear error
        clear_last_error()
        self.assertIsNone(get_last_error())
        
    def test_error_checking(self):
        """Test error checking functionality."""
        # Success should not raise
        check_error(ErrorCode.SUCCESS)
        
        # Set error and check
        error = ReadOnlyError("Cannot modify")
        set_last_error(error)
        
        with self.assertRaises(ReadOnlyError) as cm:
            check_error(ErrorCode.READONLY)
        self.assertEqual(cm.exception, error)
        
        # Error should be cleared after check
        self.assertIsNone(get_last_error())
        
        # Unknown error code should raise generic error
        with self.assertRaises(MetalFaissError) as cm:
            check_error(ErrorCode.RUNTIME)
        self.assertEqual(cm.exception.code, ErrorCode.RUNTIME)
        
    def test_error_contexts(self):
        """Test error handling in different contexts."""
        # Test read-only context
        def modify_readonly():
            raise ReadOnlyError("Cannot modify read-only index")
            
        with self.assertRaises(ReadOnlyError):
            modify_readonly()
            
        # Test memory error context
        def allocate_memory():
            raise MemoryError("Failed to allocate buffer")
            
        with self.assertRaises(MemoryError):
            allocate_memory()
            
        # Test file operations
        def read_file():
            raise FileNotFoundError("index.faiss not found")
            
        with self.assertRaises(FileNotFoundError):
            read_file()
            
        def write_file():
            raise PermissionError("Cannot write to index.faiss")
            
        with self.assertRaises(PermissionError):
            write_file()
            
    def test_error_chaining(self):
        """Test error chaining behavior."""
        try:
            try:
                raise IOError("Failed to read file")
            except IOError as e:
                raise IndexError("Cannot load index") from e
        except IndexError as e:
            self.assertIsInstance(e.__cause__, IOError)
            self.assertEqual(str(e.__cause__), "[Error IO] Failed to read file")

if __name__ == '__main__':
    unittest.main()