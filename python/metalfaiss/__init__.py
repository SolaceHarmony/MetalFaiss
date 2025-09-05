# MetalFaiss - A pure Python port of FAISS using MLX and Metal JIT acceleration
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license (see LICENSE file in root directory).

"""
MetalFaiss: A Python implementation of FAISS using Apple's MLX framework.

This package provides vector similarity search and clustering algorithms
optimized for Apple Silicon using Metal Performance Shaders through MLX.
"""

__version__ = "0.1.0"
__author__ = "Sydney Renee"
__email__ = "sydney@solace.ofharmony.ai"

# Check MLX availability
_HAS_MLX = False
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    import warnings
    warnings.warn(
        "MLX is not available. Some functionality may be limited. "
        "Install MLX with: pip install mlx",
        ImportWarning
    )

# Core classes that don't require MLX
from .metric_type import MetricType
from .search_result import SearchResult
from .search_range_result import SearchRangeResult

# Core index classes - import with fallbacks
try:
    from .index import Index
except ImportError:
    Index = None

try:
    from .indexflat import FlatIndex
except ImportError:
    FlatIndex = None

# Index implementations from index submodule
BaseIndex = None
IndexFlat = None
AnyIndex = None
IVFIndex = None
IVFFlatIndex = None
IVFScalarQuantizerIndex = None
LSHIndex = None
ScalarQuantizerIndex = None

if _HAS_MLX:
    try:
        from .index import (
            BaseIndex,
            FlatIndex as IndexFlat,
            AnyIndex,
            IVFIndex,
            IVFFlatIndex,
            IVFScalarQuantizerIndex,
            LSHIndex,
            ScalarQuantizerIndex
        )
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import index implementations: {e}", ImportWarning)

# Vector transforms
BaseVectorTransform = None
try:
    from .VectorTransform import BaseVectorTransform
except ImportError:
    pass

# Vector transform implementations
BaseLinearTransform = None
CenteringTransform = None
NormalizationTransform = None
PCAMatrixTransform = None
OPQMatrixTransform = None
ITQTransform = None
ITQMatrixTransform = None

if _HAS_MLX:
    try:
        from .vector_transform import (
            BaseLinearTransform,
            CenteringTransform,
            NormalizationTransform,
            PCAMatrixTransform,
            OPQMatrixTransform,
            ITQTransform,
            ITQMatrixTransform
        )
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import vector transforms: {e}", ImportWarning)

# Clustering
BaseClustering = None
AnyClustering = None
if _HAS_MLX:
    try:
        from .clustering import BaseClustering, AnyClustering
    except ImportError:
        pass

# Distance functions
pairwise_L2sqr = None
pairwise_extra_distances = None
if _HAS_MLX:
    try:
        from .distances import pairwise_L2sqr
        from .extra_distances import pairwise_extra_distances
    except ImportError:
        pass

# Utilities - these should work without MLX
load_data = None
encode_sentences = None
create_matrix = None
normalize_data = None
try:
    from .Utils import load_data, encode_sentences, create_matrix, normalize_data
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import utilities: {e}", ImportWarning)

# Export all available functionality
__all__ = [
    # Core classes (always available)
    "MetricType",
    "SearchResult", 
    "SearchRangeResult",
    
    # Version info
    "__version__",
    "__author__", 
    "__email__"
]

# Add available classes to __all__
_available_exports = {
    "Index": Index,
    "FlatIndex": FlatIndex,
    "BaseIndex": BaseIndex,
    "IndexFlat": IndexFlat,
    "AnyIndex": AnyIndex,
    "IVFIndex": IVFIndex,
    "IVFFlatIndex": IVFFlatIndex,
    "IVFScalarQuantizerIndex": IVFScalarQuantizerIndex,
    "LSHIndex": LSHIndex,
    "ScalarQuantizerIndex": ScalarQuantizerIndex,
    "BaseVectorTransform": BaseVectorTransform,
    "BaseLinearTransform": BaseLinearTransform,
    "CenteringTransform": CenteringTransform,
    "NormalizationTransform": NormalizationTransform,
    "PCAMatrixTransform": PCAMatrixTransform,
    "OPQMatrixTransform": OPQMatrixTransform,
    "ITQTransform": ITQTransform,
    "ITQMatrixTransform": ITQMatrixTransform,
    "BaseClustering": BaseClustering,
    "AnyClustering": AnyClustering,
    "pairwise_L2sqr": pairwise_L2sqr,
    "pairwise_extra_distances": pairwise_extra_distances,
    "load_data": load_data,
    "encode_sentences": encode_sentences,
    "create_matrix": create_matrix,
    "normalize_data": normalize_data,
}

# Only export non-None items
for name, obj in _available_exports.items():
    if obj is not None:
        __all__.append(name)

__all__ = [
    # Core classes
    "Index",
    "FlatIndex", 
    "MetricType",
    "SearchResult",
    "SearchRangeResult",
    
    # Base classes
    "BaseVectorTransform",
    
    # Utilities
    "load_data",
    "encode_sentences", 
    "create_matrix",
    "normalize_data",
    
    # Version info
    "__version__",
    "__author__",
    "__email__"
]

# Add optional exports if available
try:
    from .distances import pairwise_L2sqr
    __all__.append("pairwise_L2sqr")
except ImportError:
    pass

try:
    from .clustering import BaseClustering
    __all__.append("BaseClustering") 
except ImportError:
    pass