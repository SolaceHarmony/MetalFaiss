import pytest
import mlx.core as mx
import numpy as np
from ..transforms import (
    OPQMatrixTransform,
    RemapDimensionsTransform,
    PCAMatrixTransform,
    RandomRotationMatrixTransform
)

def create_test_data(rows=200, cols=4):
    return mx.random.uniform(0, 1, (rows, cols))

def test_opq_matrix_transform():
    data = create_test_data()
    transform = OPQMatrixTransform(d=4, m=2, d2=2)
    
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 2
    assert transform.is_trained
    assert transform.is_orthonormal

def test_pca_matrix_transform():
    data = create_test_data()
    transform = PCAMatrixTransform(d_in=4, d_out=2, eigen_power=1, random_rotation=False)
    
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 2
    assert transform.is_trained

def test_random_rotation_matrix():
    data = create_test_data()
    transform = RandomRotationMatrixTransform(d_in=4, d_out=2)
    
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 2
    assert transform.is_trained
    assert transform.is_orthonormal

def test_remap_dimensions():
    data = create_test_data()
    transform = RemapDimensionsTransform(d_in=4, d_out=2, uniform=True)
    
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 2
    assert transform.is_trained
