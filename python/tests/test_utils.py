import mlx.core as mx
import numpy as np
from typing import List

def create_test_matrix(rows: int, columns: int) -> List[List[float]]:
    """Create a test matrix with sequential values"""
    data = []
    for i in range(rows):
        data.append([float(i * columns + j) for j in range(columns)])
    return data

def assert_array_equal(a1: List[float], a2: List[float], tolerance: float = 1e-6) -> bool:
    """Compare two arrays with tolerance"""
    if len(a1) != len(a2):
        return False
    return all(abs(x - y) <= tolerance for x, y in zip(a1, a2))

def assert_matrix_equal(m1: List[List[float]], m2: List[List[float]], tolerance: float = 1e-6) -> bool:
    """Compare two matrices with tolerance"""
    if len(m1) != len(m2):
        return False
    return all(assert_array_equal(r1, r2, tolerance) for r1, r2 in zip(m1, m2))
