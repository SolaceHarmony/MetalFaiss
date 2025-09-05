"""
test_quantizer_type.py - Tests for quantizer type functionality

These tests verify that our quantizer type implementation matches FAISS,
particularly around:
- Quantizer type values
- Range statistics
- Bit widths
- Training requirements
"""

import unittest
from ..types.quantizer_type import (
    QuantizerType,
    RangeStat,
    get_bits_per_dim,
    is_uniform,
    is_direct,
    is_float,
    requires_training,
    get_range_stat_name,
    get_quantizer_name
)

class TestQuantizerType(unittest.TestCase):
    """Test quantizer type functionality."""
    
    def test_quantizer_values(self):
        """Test quantizer type values match FAISS."""
        # Basic quantization
        self.assertEqual(QuantizerType.QT_8bit, 0)
        self.assertEqual(QuantizerType.QT_4bit, 1)
        self.assertEqual(QuantizerType.QT_8bit_uniform, 2)
        self.assertEqual(QuantizerType.QT_4bit_uniform, 3)
        self.assertEqual(QuantizerType.QT_fp16, 4)
        self.assertEqual(QuantizerType.QT_8bit_direct, 5)
        self.assertEqual(QuantizerType.QT_6bit, 6)
        self.assertEqual(QuantizerType.QT_bf16, 7)
        self.assertEqual(QuantizerType.QT_8bit_direct_signed, 8)
        
    def test_range_stats(self):
        """Test range statistic values."""
        self.assertEqual(RangeStat.RS_minmax, 0)
        self.assertEqual(RangeStat.RS_meanstd, 1)
        self.assertEqual(RangeStat.RS_quantiles, 2)
        self.assertEqual(RangeStat.RS_optim, 3)
        
        # Check names
        self.assertEqual(get_range_stat_name(RangeStat.RS_minmax), "Min-Max")
        self.assertEqual(get_range_stat_name(RangeStat.RS_meanstd), "Mean-Std")
        self.assertEqual(get_range_stat_name(RangeStat.RS_quantiles), "Quantiles")
        self.assertEqual(get_range_stat_name(RangeStat.RS_optim), "Optimized")
        
    def test_bits_per_dim(self):
        """Test bit width calculations."""
        # Check all quantizer types
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_8bit), 8)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_4bit), 4)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_8bit_uniform), 8)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_4bit_uniform), 4)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_fp16), 16)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_8bit_direct), 8)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_6bit), 6)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_bf16), 16)
        self.assertEqual(get_bits_per_dim(QuantizerType.QT_8bit_direct_signed), 8)
        
    def test_uniform_check(self):
        """Test uniform range detection."""
        # Uniform quantizers
        self.assertTrue(is_uniform(QuantizerType.QT_8bit_uniform))
        self.assertTrue(is_uniform(QuantizerType.QT_4bit_uniform))
        
        # Non-uniform quantizers
        self.assertFalse(is_uniform(QuantizerType.QT_8bit))
        self.assertFalse(is_uniform(QuantizerType.QT_4bit))
        self.assertFalse(is_uniform(QuantizerType.QT_fp16))
        self.assertFalse(is_uniform(QuantizerType.QT_8bit_direct))
        self.assertFalse(is_uniform(QuantizerType.QT_6bit))
        self.assertFalse(is_uniform(QuantizerType.QT_bf16))
        self.assertFalse(is_uniform(QuantizerType.QT_8bit_direct_signed))
        
    def test_direct_check(self):
        """Test direct indexing detection."""
        # Direct quantizers
        self.assertTrue(is_direct(QuantizerType.QT_8bit_direct))
        self.assertTrue(is_direct(QuantizerType.QT_8bit_direct_signed))
        
        # Non-direct quantizers
        self.assertFalse(is_direct(QuantizerType.QT_8bit))
        self.assertFalse(is_direct(QuantizerType.QT_4bit))
        self.assertFalse(is_direct(QuantizerType.QT_8bit_uniform))
        self.assertFalse(is_direct(QuantizerType.QT_4bit_uniform))
        self.assertFalse(is_direct(QuantizerType.QT_fp16))
        self.assertFalse(is_direct(QuantizerType.QT_6bit))
        self.assertFalse(is_direct(QuantizerType.QT_bf16))
        
    def test_float_check(self):
        """Test floating point format detection."""
        # Float quantizers
        self.assertTrue(is_float(QuantizerType.QT_fp16))
        self.assertTrue(is_float(QuantizerType.QT_bf16))
        
        # Non-float quantizers
        self.assertFalse(is_float(QuantizerType.QT_8bit))
        self.assertFalse(is_float(QuantizerType.QT_4bit))
        self.assertFalse(is_float(QuantizerType.QT_8bit_uniform))
        self.assertFalse(is_float(QuantizerType.QT_4bit_uniform))
        self.assertFalse(is_float(QuantizerType.QT_8bit_direct))
        self.assertFalse(is_float(QuantizerType.QT_6bit))
        self.assertFalse(is_float(QuantizerType.QT_8bit_direct_signed))
        
    def test_training_requirements(self):
        """Test training requirement detection."""
        # Requires training
        self.assertTrue(requires_training(QuantizerType.QT_8bit))
        self.assertTrue(requires_training(QuantizerType.QT_4bit))
        self.assertTrue(requires_training(QuantizerType.QT_8bit_uniform))
        self.assertTrue(requires_training(QuantizerType.QT_4bit_uniform))
        self.assertTrue(requires_training(QuantizerType.QT_6bit))
        
        # No training required
        self.assertFalse(requires_training(QuantizerType.QT_fp16))
        self.assertFalse(requires_training(QuantizerType.QT_8bit_direct))
        self.assertFalse(requires_training(QuantizerType.QT_bf16))
        self.assertFalse(requires_training(QuantizerType.QT_8bit_direct_signed))
        
    def test_quantizer_names(self):
        """Test quantizer name formatting."""
        # Check all names
        self.assertEqual(get_quantizer_name(QuantizerType.QT_8bit), "8-bit")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_4bit), "4-bit")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_8bit_uniform), "8-bit Uniform")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_4bit_uniform), "4-bit Uniform")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_fp16), "FP16")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_8bit_direct), "8-bit Direct")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_6bit), "6-bit")
        self.assertEqual(get_quantizer_name(QuantizerType.QT_bf16), "BFloat16")
        self.assertEqual(
            get_quantizer_name(QuantizerType.QT_8bit_direct_signed),
            "8-bit Direct Signed"
        )
        
    def test_usage_examples(self):
        """Test typical usage scenarios."""
        def compute_storage_size(d: int, qtype: QuantizerType) -> int:
            """Compute storage size in bytes."""
            bits = get_bits_per_dim(qtype)
            return (d * bits + 7) // 8  # Round up to bytes
            
        # Test storage calculations
        self.assertEqual(compute_storage_size(128, QuantizerType.QT_8bit), 128)
        self.assertEqual(compute_storage_size(128, QuantizerType.QT_4bit), 64)
        self.assertEqual(compute_storage_size(128, QuantizerType.QT_6bit), 96)
        
        def select_training_method(qtype: QuantizerType, n_train: int):
            """Select training method based on quantizer type."""
            if not requires_training(qtype):
                return "No training needed"
                
            if is_uniform(qtype):
                return "Train global range"
            elif n_train < 10000:
                return "Use min-max per dimension"
            else:
                return "Use optimized per dimension"
                
        # Test training selection
        self.assertEqual(
            select_training_method(QuantizerType.QT_fp16, 1000),
            "No training needed"
        )
        self.assertEqual(
            select_training_method(QuantizerType.QT_8bit_uniform, 1000),
            "Train global range"
        )
        self.assertEqual(
            select_training_method(QuantizerType.QT_8bit, 1000),
            "Use min-max per dimension"
        )
        self.assertEqual(
            select_training_method(QuantizerType.QT_8bit, 100000),
            "Use optimized per dimension"
        )

if __name__ == '__main__':
    unittest.main()