from typing import Optional

class IndexError(Exception):
    """Base exception class for index-related errors in MetalFaiss"""
    
    def __init__(self, message: str, result_code: int = -1):
        """
        Initialize IndexError.
        
        Args:
            message: Error message
            result_code: Optional error code (defaults to -1)
        """
        super().__init__(message)
        self.result_code = result_code
        self.message = message
        
    @classmethod
    def check(cls, result_code: int, message: Optional[str] = None) -> None:
        """
        Check result code and raise IndexError if non-zero.
        
        Args:
            result_code: Error code to check
            message: Optional error message
            
        Raises:
            IndexError: If result_code is non-zero
        """
        if result_code != 0:
            raise cls(
                message or f"Operation failed with code {result_code}", 
                result_code
            )
            
    def __str__(self) -> str:
        """String representation including both message and code"""
        return f"{self.message} (code: {self.result_code})"
