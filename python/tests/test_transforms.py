import pytest
import mlx.core as mx
import numpy as np
from ..vector_transform import (
    CenteringTransform,
    ITQMatrixTransform,
    NormalizationTransform,
    OPQMatrixTransform,
    PCAMatrixTransform,
    RandomRotationMatrixTransform,
    RemapDimensionsTransform,
    itq_transform
)

def create_test_data(rows=200, columns=4):
    return mx.random.uniform(0, 1, (rows, columns))

def test_centering_transform():
    data = create_test_data()
    transform = CenteringTransform(d=4)
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 4
    assert transform.is_trained
    
def test_itq_matrix_transform():
    data = create_test_data()
    transform = ITQMatrixTransform(d=4)
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 4
    assert transform.is_trained
    assert not transform.is_orthonormal

# ...existing test cases for other transforms...

def test_normalization_transform():
    data = create_test_data()
    transform = NormalizationTransform(d=4, norm=2.0)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 4
    assert transform.is_trained

def test_pca_matrix_transform():
    data = create_test_data()
    transform = PCAMatrixTransform(d_in=4, d_out=2, eigen_power=1, random_rotation=False)
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 2
    assert transform.is_trained

def test_itq_transform():
    data = create_test_data()
    transform = itq_transform(d_in=4, d_out=2, do_pca=True)
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 2
    assert transform.is_trained
