from typing import Optional
import mlx.core as mx
from .base_index import BaseIndex
from ..metric_type import MetricType
from ..errors import IndexError

class IVFIndex(BaseIndex):
    """Base class for all IVF (Inverted File) indexes.
    
    This class provides the Python equivalent of the Swift IVFIndex implementation,
    adapted to use MLX for GPU acceleration where possible.
    """
    
    def __init__(self, index_pointer: Optional['IVFIndex'] = None):
        """Initialize IVF index.
        
        Args:
            index_pointer: Optional existing IVFIndex to wrap
        """
        if index_pointer is not None:
            super().__init__(index_pointer.d)
            self._nprobe = index_pointer.nprobe
            self._nlist = index_pointer.nlist
            self._metric_type = index_pointer.metric_type
            self._direct_map = False
            self._index_pointer = index_pointer
        else:
            super().__init__(0)
            self._nprobe = 1
            self._nlist = 0
            self._direct_map = False
            self._index_pointer = self

    @property
    def index_pointer(self) -> 'IVFIndex':
        """Get underlying index pointer."""
        return self._index_pointer

    @staticmethod
    def from_(index_pointer: Optional['IVFIndex']) -> Optional['IVFIndex']:
        """Create from index pointer for Swift compatibility.

        Args:
            index_pointer: Index to convert

        Returns:
            IVFIndex if conversion possible, None otherwise
        """
        return index_pointer if isinstance(index_pointer, IVFIndex) else None

    @property
    def nprobe(self) -> int:
        """Number of nearest cells to probe during search."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        """Set number of cells to probe during search.
        
        Args:
            value: Number of cells to probe
            
        Raises:
            ValueError: If value is less than 1
        """
        if value < 1:
            raise ValueError("nprobe must be positive")
        self._nprobe = value

    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist

    @property
    def imbalance_factor(self) -> float:
        """Calculate imbalance factor of inverted lists."""
        if not hasattr(self, '_invlists'):
            return 0.0
        sizes = [len(l) for l in self._invlists]
        if not sizes:
            return 0.0
        avg = sum(sizes) / len(sizes)
        if avg == 0:
            return 0.0
        sqsum = sum(s*s for s in sizes)
        return sqsum / (avg * avg * len(sizes))

    def make_direct_map(self) -> None:
        """Enable direct mapping of vectors to lists.
        
        Raises:
            IndexError: If operation fails
        """
        self._direct_map = True

    def clear_direct_map(self) -> None:
        """Disable direct mapping of vectors to lists.
        
        Raises:
            IndexError: If operation fails
        """
        self._direct_map = False
