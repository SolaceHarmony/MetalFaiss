# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

from enum import Enum
from typing import Optional

class MetricType(Enum):
    """Available distance metric types for index comparison."""
    
    INNER_PRODUCT = 0
    L2 = 1
    L1 = 2
    LINF = 3
    LP = 4
    CANBERRA = 5
    BRAY_CURTIS = 6
    JENSEN_SHANNON = 7
    
    @classmethod
    def from_string(cls, metric_name: str) -> 'MetricType':
        """
        Convert string representation to MetricType.
        
        Args:
            metric_name: String name of metric (case insensitive)
            
        Returns:
            Corresponding MetricType
            
        Raises:
            ValueError: If metric name is not recognized
        """
        mapping = {
            'inner_product': cls.INNER_PRODUCT,
            'l2': cls.L2,
            'l1': cls.L1,
            'linf': cls.LINF,
            'lp': cls.LP,
            'canberra': cls.CANBERRA,
            'bray_curtis': cls.BRAY_CURTIS,
            'jensen_shannon': cls.JENSEN_SHANNON
        }
        try:
            return mapping[metric_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown metric type: {metric_name}")
            
    def __str__(self) -> str:
        """Return string representation of metric type."""
        return self.name.lower()