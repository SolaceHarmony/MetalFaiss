"""
test_metric_type.py - Tests for metric type functionality

These tests verify that our metric type implementation matches FAISS,
particularly around:
- Metric type values
- Similarity vs distance metrics
- Metric argument requirements
- Error handling
"""

import unittest
from ..types.metric_type import (
    MetricType,
    SIMILARITY_METRICS,
    is_similarity_metric,
    requires_metric_arg,
    check_metric_type,
    get_metric_name
)
from ..errors import InvalidArgumentError

class TestMetricType(unittest.TestCase):
    """Test metric type functionality."""
    
    def test_metric_values(self):
        """Test metric type values match FAISS."""
        # Core metrics
        self.assertEqual(MetricType.INNER_PRODUCT, 0)
        self.assertEqual(MetricType.L2, 1)
        self.assertEqual(MetricType.L1, 2)
        self.assertEqual(MetricType.LINF, 3)
        self.assertEqual(MetricType.LP, 4)
        
        # Additional metrics
        self.assertEqual(MetricType.CANBERRA, 20)
        self.assertEqual(MetricType.BRAY_CURTIS, 21)
        self.assertEqual(MetricType.JENSEN_SHANNON, 22)
        self.assertEqual(MetricType.JACCARD, 23)
        self.assertEqual(MetricType.NAN_EUCLIDEAN, 24)
        self.assertEqual(MetricType.ABS_INNER_PRODUCT, 25)
        
    def test_similarity_metrics(self):
        """Test similarity metric identification."""
        # Check similarity metrics
        self.assertTrue(is_similarity_metric(MetricType.INNER_PRODUCT))
        self.assertTrue(is_similarity_metric(MetricType.JACCARD))
        self.assertTrue(is_similarity_metric(MetricType.ABS_INNER_PRODUCT))
        
        # Check distance metrics
        self.assertFalse(is_similarity_metric(MetricType.L2))
        self.assertFalse(is_similarity_metric(MetricType.L1))
        self.assertFalse(is_similarity_metric(MetricType.LINF))
        self.assertFalse(is_similarity_metric(MetricType.LP))
        self.assertFalse(is_similarity_metric(MetricType.CANBERRA))
        self.assertFalse(is_similarity_metric(MetricType.BRAY_CURTIS))
        self.assertFalse(is_similarity_metric(MetricType.JENSEN_SHANNON))
        self.assertFalse(is_similarity_metric(MetricType.NAN_EUCLIDEAN))
        
        # Verify SIMILARITY_METRICS set
        self.assertEqual(
            SIMILARITY_METRICS,
            {
                MetricType.INNER_PRODUCT,
                MetricType.JACCARD,
                MetricType.ABS_INNER_PRODUCT
            }
        )
        
    def test_metric_arguments(self):
        """Test metric argument requirements."""
        # Only LP metric requires argument
        self.assertTrue(requires_metric_arg(MetricType.LP))
        
        # Other metrics don't require arguments
        for metric in MetricType:
            if metric != MetricType.LP:
                self.assertFalse(requires_metric_arg(metric))
                
    def test_metric_validation(self):
        """Test metric type validation."""
        # Valid metrics should pass
        for metric in MetricType:
            check_metric_type(metric)
            
        # Invalid metrics should raise
        with self.assertRaises(ValueError):
            check_metric_type(100)  # Invalid int
            
        with self.assertRaises(ValueError):
            check_metric_type("L2")  # Invalid string
            
    def test_metric_names(self):
        """Test metric name formatting."""
        # Check all metric names
        self.assertEqual(get_metric_name(MetricType.INNER_PRODUCT), "Inner Product")
        self.assertEqual(get_metric_name(MetricType.L2), "L2")
        self.assertEqual(get_metric_name(MetricType.L1), "L1")
        self.assertEqual(get_metric_name(MetricType.LINF), "L∞")
        self.assertEqual(get_metric_name(MetricType.LP), "Lp")
        self.assertEqual(get_metric_name(MetricType.CANBERRA), "Canberra")
        self.assertEqual(get_metric_name(MetricType.BRAY_CURTIS), "Bray-Curtis")
        self.assertEqual(get_metric_name(MetricType.JENSEN_SHANNON), "Jensen-Shannon")
        self.assertEqual(get_metric_name(MetricType.JACCARD), "Jaccard")
        self.assertEqual(get_metric_name(MetricType.NAN_EUCLIDEAN), "NaN-Euclidean")
        self.assertEqual(
            get_metric_name(MetricType.ABS_INNER_PRODUCT),
            "Absolute Inner Product"
        )
        
    def test_metric_usage(self):
        """Test metric type usage in typical scenarios."""
        # Test similarity vs distance handling
        def sort_results(metric: MetricType, scores: list[float]) -> list[float]:
            """Sort scores based on metric type."""
            return sorted(
                scores,
                reverse=is_similarity_metric(metric)
            )
            
        # For similarity metrics, higher scores are better
        scores = [0.1, 0.5, 0.8, 0.3]
        self.assertEqual(
            sort_results(MetricType.INNER_PRODUCT, scores),
            [0.8, 0.5, 0.3, 0.1]  # Descending
        )
        
        # For distance metrics, lower scores are better
        self.assertEqual(
            sort_results(MetricType.L2, scores),
            [0.1, 0.3, 0.5, 0.8]  # Ascending
        )
        
        # Test LP metric with argument
        def compute_lp_distance(p: float) -> None:
            """Compute Lp distance."""
            if not requires_metric_arg(MetricType.LP):
                raise ValueError("LP metric requires p argument")
            # Would compute distance here...
            
        # Should work with LP metric
        compute_lp_distance(2.0)  # p=2 for L2 distance
        
        # Test Jaccard input validation
        def compute_jaccard(vectors: list[float]) -> None:
            """Compute Jaccard similarity."""
            if any(x < 0 for x in vectors):
                raise ValueError("Jaccard requires non-negative vectors")
            # Would compute similarity here...
            
        # Should work with non-negative vectors
        compute_jaccard([0.1, 0.2, 0.3])
        
        # Should fail with negative vectors
        with self.assertRaises(ValueError):
            compute_jaccard([0.1, -0.2, 0.3])

if __name__ == '__main__':
    unittest.main()