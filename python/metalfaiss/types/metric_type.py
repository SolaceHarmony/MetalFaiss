"""
metric_type.py - Distance metric types for MetalFaiss
"""

from enum import Enum
from typing import List, Optional, Union

class MetricType(Enum):
    """Distance metric types."""
    L2 = "L2"                          # Euclidean distance
    INNER_PRODUCT = "INNER_PRODUCT"    # Inner product (a.k.a. dot product)
    L1 = "L1"                          # Manhattan distance
    Linf = "Linf"                      # Maximum coordinate difference
    Lp = "Lp"                          # Minkowski distance with custom p
    Canberra = "Canberra"              # Canberra distance
    BrayCurtis = "BrayCurtis"          # Bray-Curtis distance
    JensenShannon = "JensenShannon"    # Jensen-Shannon divergence
    Jaccard = "Jaccard"                # Jaccard distance

# Constants for backward compatibility
METRIC_INNER_PRODUCT = MetricType.INNER_PRODUCT
METRIC_L2 = MetricType.L2
METRIC_L1 = MetricType.L1
METRIC_Linf = MetricType.Linf
METRIC_Lp = MetricType.Lp
METRIC_Canberra = MetricType.Canberra
METRIC_BrayCurtis = MetricType.BrayCurtis
METRIC_JensenShannon = MetricType.JensenShannon
METRIC_Jaccard = MetricType.Jaccard

# Lists of metric types by category
SIMILARITY_METRICS = [
    MetricType.INNER_PRODUCT
]

DISTANCE_METRICS = [
    MetricType.L2,
    MetricType.L1,
    MetricType.Linf,
    MetricType.Lp,
    MetricType.Canberra,
    MetricType.BrayCurtis,
    MetricType.JensenShannon,
    MetricType.Jaccard
]

SYMMETRIC_METRICS = [
    MetricType.L2,
    MetricType.L1,
    MetricType.Linf,
    MetricType.Lp,
    MetricType.Canberra,
    MetricType.BrayCurtis,
    MetricType.JensenShannon,
    MetricType.Jaccard
]

NORMALIZED_METRICS = [
    MetricType.INNER_PRODUCT,
    MetricType.Jaccard,
    MetricType.JensenShannon
]

METRICS_WITH_ARGS = [
    MetricType.Lp  # Requires p parameter
]

def is_similarity_metric(metric: MetricType) -> bool:
    """Check if metric is a similarity metric (higher is better)."""
    return metric in SIMILARITY_METRICS

def is_distance_metric(metric: MetricType) -> bool:
    """Check if metric is a distance metric (lower is better)."""
    return metric in DISTANCE_METRICS

def metric_is_symmetric(metric: MetricType) -> bool:
    """Check if metric is symmetric."""
    return metric in SYMMETRIC_METRICS

def metric_needs_normalization(metric: MetricType) -> bool:
    """Check if metric needs vector normalization."""
    return metric in NORMALIZED_METRICS

def requires_metric_arg(metric: MetricType) -> bool:
    """Check if metric requires additional argument.
    
    Args:
        metric: Metric type to check
        
    Returns:
        True if metric requires additional argument (e.g., p for Lp)
    """
    return metric in METRICS_WITH_ARGS

def get_metric_name(metric: MetricType) -> str:
    """Get string name of metric type.
    
    Args:
        metric: Metric type
        
    Returns:
        String name of metric
    """
    return metric.value

def get_metric_description(metric: MetricType) -> str:
    """Get description of metric type.
    
    Args:
        metric: Metric type
        
    Returns:
        Description of metric
    """
    descriptions = {
        MetricType.L2: "Euclidean distance",
        MetricType.INNER_PRODUCT: "Inner product (dot product)",
        MetricType.L1: "Manhattan distance",
        MetricType.Linf: "Maximum coordinate difference",
        MetricType.Lp: "Minkowski distance with custom p",
        MetricType.Canberra: "Canberra distance",
        MetricType.BrayCurtis: "Bray-Curtis distance",
        MetricType.JensenShannon: "Jensen-Shannon divergence",
        MetricType.Jaccard: "Jaccard distance"
    }
    return descriptions[metric]

def check_metric_type(
    metric: Union[MetricType, str],
    metric_arg: Optional[float] = None
) -> MetricType:
    """Validate and convert metric type.
    
    Args:
        metric: Metric type or string name
        metric_arg: Optional metric argument (e.g., p for Lp)
        
    Returns:
        Validated MetricType enum
        
    Raises:
        ValueError: If metric is invalid or missing required argument
    """
    # Convert string to enum
    if isinstance(metric, str):
        try:
            metric = MetricType(metric)
        except ValueError:
            raise ValueError(f"Unknown metric type: {metric}")
            
    # Check if metric is valid
    if not isinstance(metric, MetricType):
        raise ValueError(f"Invalid metric type: {metric}")
        
    # Check if metric argument is provided when needed
    if requires_metric_arg(metric) and metric_arg is None:
        raise ValueError(f"Metric {metric} requires argument")
        
    # Check if metric argument is valid
    if metric == MetricType.Lp and metric_arg is not None:
        if metric_arg <= 0:
            raise ValueError(f"Invalid p value for Lp metric: {metric_arg}")
            
    return metric

def get_default_metric_arg(metric: MetricType) -> Optional[float]:
    """Get default argument for metric type.
    
    Args:
        metric: Metric type
        
    Returns:
        Default argument value or None if not needed
    """
    if metric == MetricType.Lp:
        return 2.0  # Default to L2 norm
    return None