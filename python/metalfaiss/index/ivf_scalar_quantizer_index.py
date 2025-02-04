import mlx.core as mx
from typing import Optional, Union
from ..index_pointer import IndexPointer
from .base_index import BaseIndex
from ..metric_type import MetricType
from ..quantizer_type import QuantizerType

class IVFScalarQuantizerIndex(BaseIndex):
    def __init__(self, index_pointer: IndexPointer, quantizer: Optional[BaseIndex] = None):
        self._index_pointer = index_pointer
        self._quantizer = quantizer
        self._d = 0  # Will be set from index_pointer
        self._nlist = 0
        self._nprobe = 1
        self._metric_type = None
        
    def __del__(self):
        if hasattr(self, '_index_pointer') and self._index_pointer is not None:
            # Clean up if we're the last reference
            self._index_pointer = None

    @staticmethod
    def from_(index_pointer: Optional[IndexPointer]) -> Optional['IVFScalarQuantizerIndex']:
        """Create from index pointer for Swift compatibility."""
        if index_pointer is None:
            return None
        # Verify the pointer is valid for IVFScalarQuantizer type
        if not index_pointer.is_valid:
            return None
        return IVFScalarQuantizerIndex(index_pointer=index_pointer, quantizer=None)

    @classmethod 
    def create(cls,
              quantizer: BaseIndex,
              quantizer_type: QuantizerType,
              d: int,
              nlist: int,
              metric_type: MetricType = MetricType.L2,
              encode_residual: bool = True) -> 'IVFScalarQuantizerIndex':
        """Convenience initializer matching Swift's implementation."""
        # Create new index pointer using MLX/Metal backend
        index_pointer = IndexPointer(None)  # TODO: Implement actual MLX/Metal initialization
        index = cls(index_pointer=index_pointer, quantizer=quantizer)
        index._d = d
        index._nlist = nlist
        index._metric_type = metric_type
        return index

    @property
    def index_pointer(self) -> IndexPointer:
        """Get the underlying index pointer. Matches Swift's public internal(set)."""
        return self._index_pointer
    
    # No setter for index_pointer to match Swift's internal(set)

    @property
    def nprobe(self) -> int:
        """Number of nearest cells to probe during search."""
        return self._nprobe
        
    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value < 1:
            raise ValueError("nprobe must be positive")
        self._nprobe = value

    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist
