import mlx.core as mx
import numpy as np

class SearchRangeResult:
    def __init__(self, lims, distances, labels):
        self.lims = lims  # List[int] - index limits in distances/labels for each query
        self.distances = distances  # List[float] - distances for matches within radius
        self.labels = labels  # List[int] - labels for matches within radius

def search_range(vectors, query, radius):
    vectors = mx.eval(mx.array(vectors, dtype=mx.float32))
    query = mx.eval(mx.array(query, dtype=mx.float32))
    
    # Compute distances and force evaluation
    distances = mx.eval(mx.linalg.norm(
        query[:, None, :] - vectors[None, :, :],
        axis=2
    ))
    
    matches = mx.eval(distances <= radius)
    
    # Build result
    lims = []
    all_distances = []
    all_labels = []
    
    offset = 0
    for i in range(len(query)):
        matches_i = mx.where(matches[i])[0]
        lims.append(offset + len(matches_i))
        all_distances.extend(distances[i][matches_i])
        all_labels.extend(matches_i)
        offset = lims[-1]
        
    return SearchRangeResult(lims, all_distances, all_labels)
