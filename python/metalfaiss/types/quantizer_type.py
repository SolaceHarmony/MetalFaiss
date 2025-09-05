"""
quantizer_type.py - Quantizer types for MetalFaiss
"""

from enum import Enum
from typing import List, Optional, Union

class QuantizerType(Enum):
    """Scalar quantizer types."""
    QT_8bit = "8bit"                  # 8-bit scalar quantization
    QT_4bit = "4bit"                  # 4-bit scalar quantization
    QT_8bit_uniform = "8bit_uniform"  # 8-bit uniform scalar quantization
    QT_4bit_uniform = "4bit_uniform"  # 4-bit uniform scalar quantization
    QT_fp16 = "fp16"                  # 16-bit float quantization
    QT_8bit_direct = "8bit_direct"    # Direct 8-bit quantization
    QT_6bit = "6bit"                  # 6-bit scalar quantization

class RangeStat(Enum):
    """Range statistics for quantizer training."""
    RS_minmax = "minmax"        # Use min/max statistics
    RS_meanstd = "meanstd"      # Use mean/std statistics
    RS_quantiles = "quantiles"  # Use quantile statistics
    RS_optim = "optim"          # Use optimal clustering

# Constants for backward compatibility
QT_8bit = QuantizerType.QT_8bit
QT_4bit = QuantizerType.QT_4bit
QT_8bit_uniform = QuantizerType.QT_8bit_uniform
QT_4bit_uniform = QuantizerType.QT_4bit_uniform
QT_fp16 = QuantizerType.QT_fp16
QT_8bit_direct = QuantizerType.QT_8bit_direct
QT_6bit = QuantizerType.QT_6bit

RS_minmax = RangeStat.RS_minmax
RS_meanstd = RangeStat.RS_meanstd
RS_quantiles = RangeStat.RS_quantiles
RS_optim = RangeStat.RS_optim

# Lists of quantizer types by category
TRAINED_QUANTIZERS = [
    QuantizerType.QT_8bit,
    QuantizerType.QT_4bit,
    QuantizerType.QT_6bit
]

UNIFORM_QUANTIZERS = [
    QuantizerType.QT_8bit_uniform,
    QuantizerType.QT_4bit_uniform,
    QuantizerType.QT_8bit_direct
]

FLOAT_QUANTIZERS = [
    QuantizerType.QT_fp16
]

DIRECT_QUANTIZERS = [
    QuantizerType.QT_8bit_direct
]

# Lists of range stats by category
SIMPLE_RANGE_STATS = [
    RangeStat.RS_minmax,
    RangeStat.RS_meanstd
]

COMPLEX_RANGE_STATS = [
    RangeStat.RS_quantiles,
    RangeStat.RS_optim
]

def get_bits_per_code(qtype: QuantizerType) -> int:
    """Get number of bits per code for quantizer type.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        Number of bits per code
    """
    if qtype == QuantizerType.QT_8bit:
        return 8
    elif qtype == QuantizerType.QT_4bit:
        return 4
    elif qtype == QuantizerType.QT_8bit_uniform:
        return 8
    elif qtype == QuantizerType.QT_4bit_uniform:
        return 4
    elif qtype == QuantizerType.QT_fp16:
        return 16
    elif qtype == QuantizerType.QT_8bit_direct:
        return 8
    elif qtype == QuantizerType.QT_6bit:
        return 6
    else:
        raise ValueError(f"Unknown quantizer type: {qtype}")

def get_bits_per_dim(qtype: QuantizerType, d: int, M: Optional[int] = None) -> float:
    """Get number of bits per dimension.
    
    Args:
        qtype: Quantizer type
        d: Vector dimension
        M: Number of sub-quantizers (for PQ)
        
    Returns:
        Number of bits per dimension
    """
    bits = get_bits_per_code(qtype)
    if M is not None:
        # Product quantizer - bits are split across dimensions
        return bits * M / d
    else:
        # Scalar quantizer - same bits for each dimension
        return float(bits)

def needs_training(qtype: QuantizerType) -> bool:
    """Check if quantizer type needs training.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        True if quantizer needs training
    """
    return qtype in TRAINED_QUANTIZERS

def requires_training(qtype: QuantizerType) -> bool:
    """Alias for needs_training."""
    return needs_training(qtype)

def is_uniform(qtype: QuantizerType) -> bool:
    """Check if quantizer type is uniform.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        True if quantizer is uniform
    """
    return qtype in UNIFORM_QUANTIZERS

def is_float(qtype: QuantizerType) -> bool:
    """Check if quantizer type uses floating point.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        True if quantizer uses floating point
    """
    return qtype in FLOAT_QUANTIZERS

def is_direct(qtype: QuantizerType) -> bool:
    """Check if quantizer type uses direct encoding.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        True if quantizer uses direct encoding
    """
    return qtype in DIRECT_QUANTIZERS

def is_simple_range_stat(stat: RangeStat) -> bool:
    """Check if range stat is simple.
    
    Args:
        stat: Range stat type
        
    Returns:
        True if range stat is simple
    """
    return stat in SIMPLE_RANGE_STATS

def is_complex_range_stat(stat: RangeStat) -> bool:
    """Check if range stat is complex.
    
    Args:
        stat: Range stat type
        
    Returns:
        True if range stat is complex
    """
    return stat in COMPLEX_RANGE_STATS

def get_quantizer_name(qtype: QuantizerType) -> str:
    """Get string name of quantizer type.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        String name of quantizer
    """
    return qtype.value

def get_range_stat_name(stat: RangeStat) -> str:
    """Get string name of range stat type.
    
    Args:
        stat: Range stat type
        
    Returns:
        String name of range stat
    """
    return stat.value

def get_default_range_stat(qtype: QuantizerType) -> RangeStat:
    """Get default range stat for quantizer type.
    
    Args:
        qtype: Quantizer type
        
    Returns:
        Default range stat
    """
    if qtype in TRAINED_QUANTIZERS:
        return RangeStat.RS_optim
    else:
        return RangeStat.RS_minmax

def check_quantizer_type(
    qtype: Union[QuantizerType, str],
    d: Optional[int] = None,
    M: Optional[int] = None
) -> QuantizerType:
    """Validate and convert quantizer type.
    
    Args:
        qtype: Quantizer type or string name
        d: Optional vector dimension
        M: Optional number of sub-quantizers
        
    Returns:
        Validated QuantizerType enum
        
    Raises:
        ValueError: If quantizer type is invalid or incompatible with d/M
    """
    # Convert string to enum
    if isinstance(qtype, str):
        try:
            qtype = QuantizerType(qtype)
        except ValueError:
            raise ValueError(f"Unknown quantizer type: {qtype}")
            
    # Check if quantizer is valid
    if not isinstance(qtype, QuantizerType):
        raise ValueError(f"Invalid quantizer type: {qtype}")
        
    # Check dimension compatibility
    if d is not None and M is not None:
        if d % M != 0:
            raise ValueError(f"Dimension {d} must be divisible by M={M}")
            
    return qtype
