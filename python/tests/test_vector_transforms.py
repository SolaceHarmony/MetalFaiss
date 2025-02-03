import pytest
import mlx.core as mx
import numpy as np
from ..VectorTransforms import *

def create_test_data(rows=200, cols=4):
    return mx.random.uniform(0, 1, (rows, cols))

def test_centering_transform():
    data = create_test_data()
    transform = CenteringTransform(d=4)
    
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 4  
    assert transform.is_trained
    
    # Test mean is approximately zero after centering
    assert mx.mean(mx.abs(mx.mean(result, axis=0))) < 1e-6

def test_itq_matrix_transform():
    data = create_test_data()
    transform = ITQMatrixTransform(d=4)
    
    transform.train(data)
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 4
    assert transform.is_trained
    assert not transform.is_orthonormal

    # Test reversibility
    recovered = transform.reverse_transform(result)
    assert mx.mean(mx.abs(data - recovered)) < 1e-6

def test_normalization_transform():
    data = create_test_data() 
    transform = NormalizationTransform(d=4, norm=2.0)
    
    result = transform.apply(data)
    
    assert transform.d_in == 4
    assert transform.d_out == 4
    assert transform.is_trained
    
    # Test vectors are normalized to specified norm
    norms = mx.linalg.norm(result, axis=1)
    assert mx.mean(mx.abs(norms - 2.0)) < 1e-6
