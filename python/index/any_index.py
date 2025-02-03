from typing import Optional

from .base_index import BaseIndex
from .flat_index import FlatIndex
from .ivf_index import IVFIndex
from .ivf_flat_index import IVFFlatIndex
from .ivf_scalar_quantizer_index import IVFScalarQuantizerIndex
from .lsh_index import LSHIndex
from .scalar_quantizer_index import ScalarQuantizerIndex
from .id_map import IDMap, IDMap2
from .refine_flat_index import RefineFlatIndex
from ..metric_type import MetricType

class AnyIndex(BaseIndex):
    """Generic index that can be converted to specific index types."""
    
    def __init__(self, d: int, metric_type: MetricType, description: Optional[str] = None):
        """Initialize generic index.
        
        Args:
            d: Dimension of vectors to index
            metric_type: Type of distance metric to use
            description: Optional index factory description string
        """
        super().__init__(d)
        self.metric_type = metric_type
        if description:
            self._create_from_description(description)

    @staticmethod
    def PQ(d: int, m: int, nbit: int = 8, metric_type: MetricType = MetricType.L2) -> 'AnyIndex':
        """Create Product Quantization index."""
        return AnyIndex(d, metric_type, description=f"PQ{m}x{nbit}")

    @staticmethod
    def IVFPQ(d: int, m: int, nbit: int = 8, metric_type: MetricType = MetricType.L2) -> 'AnyIndex':
        """Create IVF Product Quantization index."""
        return AnyIndex(d, metric_type, description=f"IVF{m},PQ{nbit}")

    def to_flat(self) -> Optional[FlatIndex]:
        """Convert to FlatIndex if possible."""
        return FlatIndex.from_index(self) if isinstance(self, FlatIndex) else None

    def to_ivf(self) -> Optional[IVFIndex]:
        """Convert to IVFIndex if possible."""
        return IVFIndex.from_index(self) if isinstance(self, IVFIndex) else None

    def to_ivf_flat(self) -> Optional[IVFFlatIndex]:
        """Convert to IVFFlatIndex if possible."""
        return IVFFlatIndex.from_index(self) if isinstance(self, IVFFlatIndex) else None

    def to_ivf_scalar_quantizer(self) -> Optional[IVFScalarQuantizerIndex]:
        """Convert to IVFScalarQuantizerIndex if possible."""
        return IVFScalarQuantizerIndex.from_index(self) if isinstance(self, IVFScalarQuantizerIndex) else None

    def to_lsh(self) -> Optional[LSHIndex]:
        """Convert to LSHIndex if possible."""
        return LSHIndex.from_index(self) if isinstance(self, LSHIndex) else None

    def to_id_map(self) -> Optional[IDMap]:
        """Convert to IDMap if possible."""
        return IDMap.from_index(self) if isinstance(self, IDMap) else None

    def to_id_map2(self) -> Optional[IDMap2]:
        """Convert to IDMap2 if possible."""
        return IDMap2.from_index(self) if isinstance(self, IDMap2) else None

    def to_refine_flat(self) -> Optional[RefineFlatIndex]:
        """Convert to RefineFlatIndex if possible."""
        return RefineFlatIndex.from_index(self) if isinstance(self, RefineFlatIndex) else None

    def to_scalar_quantizer(self) -> Optional[ScalarQuantizerIndex]:
        """Convert to ScalarQuantizerIndex if possible."""
        return ScalarQuantizerIndex.from_index(self) if isinstance(self, ScalarQuantizerIndex) else None
