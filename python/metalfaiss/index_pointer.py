from typing import Any
import mlx.core as mx

class IndexPointer:
    """Wrapper class for managing index pointers."""
    
    def __init__(self, pointer: Any):
        """Initialize with a pointer to an index.
        
        Args:
            pointer: Raw pointer/handle to the index
        """
        self._pointer = pointer
        
    @property
    def pointer(self) -> Any:
        """Get the underlying pointer. Similar to Swift's pointer property."""
        return self._pointer
        
    @property
    def is_valid(self) -> bool:
        """Check if the pointer is valid and of correct type."""
        # TODO: Implement proper MLX/Metal type checking here
        return self._pointer is not None
