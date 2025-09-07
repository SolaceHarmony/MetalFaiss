"""
test_id_selector.py - Tests for ID selection functionality

These tests verify that our ID selector implementations match the behavior
of the original FAISS implementations, particularly around:
- Range selection
- Bitmap operations
- Bloom filter optimization
- Logical operations
"""

import unittest
import mlx.core as mx
from ..index.id_selector import (
    IDSelector,
    IDSelectorRange,
    IDSelectorArray,
    IDSelectorBatch,
    IDSelectorBitmap,
    IDSelectorAll,
    IDSelectorNot,
    IDSelectorAnd,
    IDSelectorOr,
    IDSelectorXor
)

class TestIDSelector(unittest.TestCase):
    """Test ID selector implementations."""
    
    def test_range_selector(self):
        """Test range-based selection."""
        # Basic range
        selector = IDSelectorRange(5, 10)
        self.assertFalse(selector.is_member(4))
        self.assertTrue(selector.is_member(5))
        self.assertTrue(selector.is_member(7))
        self.assertTrue(selector.is_member(9))
        self.assertFalse(selector.is_member(10))
        
        # Empty range
        selector = IDSelectorRange(5, 5)
        self.assertFalse(selector.is_member(4))
        self.assertFalse(selector.is_member(5))
        
        # Sorted ID bounds
        selector = IDSelectorRange(5, 10, assume_sorted=True)
        ids = np.array([1, 3, 5, 6, 8, 9, 11, 12])
        jmin, jmax = selector.find_sorted_ids_bounds(len(ids), ids)
        self.assertEqual(jmin, 2)  # Index of 5
        self.assertEqual(jmax, 6)  # Index after 9
        
    def test_array_selector(self):
        """Test array-based selection."""
        ids = [1, 5, 10, 15]
        selector = IDSelectorArray(ids)
        
        # Test membership
        self.assertTrue(selector.is_member(1))
        self.assertTrue(selector.is_member(10))
        self.assertFalse(selector.is_member(2))
        self.assertFalse(selector.is_member(20))
        
        # Test with duplicates
        selector = IDSelectorArray([1, 1, 5, 5])
        self.assertTrue(selector.is_member(1))
        self.assertTrue(selector.is_member(5))
        self.assertFalse(selector.is_member(2))
        
    def test_batch_selector(self):
        """Test batch selection with Bloom filter."""
        # Create large batch of IDs
        n = 1000
        ids = list(range(0, 3*n, 3))  # [0, 3, 6, ...]
        selector = IDSelectorBatch(ids)
        
        # Test membership
        for i in range(n):
            if i % 3 == 0:
                self.assertTrue(selector.is_member(i))
            else:
                self.assertFalse(selector.is_member(i))
                
        # Test Bloom filter effectiveness
        # Most non-members should be rejected by Bloom filter
        bloom_rejects = 0
        for i in range(10000):
            if not selector.is_member(i):
                # Check if Bloom filter bits are set
                h = i & selector.mask
                if not (selector.bloom[h >> 3] & (1 << (h & 7))):
                    bloom_rejects += 1
                    
        # At least 50% of non-members should be rejected by Bloom filter
        self.assertGreater(bloom_rejects / 10000, 0.5)
        
    def test_bitmap_selector(self):
        """Test bitmap-based selection."""
        # Create bitmap selecting every other ID
        n = 16
        bitmap = np.zeros(2, dtype=np.uint8)  # 16 bits
        for i in range(0, n, 2):
            bitmap[i >> 3] |= 1 << (i & 7)
            
        selector = IDSelectorBitmap(n, bitmap)
        
        # Test membership
        for i in range(n):
            if i % 2 == 0:
                self.assertTrue(selector.is_member(i))
            else:
                self.assertFalse(selector.is_member(i))
                
        # Test out of range
        self.assertFalse(selector.is_member(-1))
        self.assertFalse(selector.is_member(n))
        
    def test_all_selector(self):
        """Test select-all selector."""
        selector = IDSelectorAll()
        for i in range(-10, 10):
            self.assertTrue(selector.is_member(i))
            
    def test_not_selector(self):
        """Test NOT operation."""
        base = IDSelectorRange(5, 10)
        selector = IDSelectorNot(base)
        
        self.assertTrue(selector.is_member(4))
        self.assertFalse(selector.is_member(5))
        self.assertFalse(selector.is_member(7))
        self.assertTrue(selector.is_member(10))
        
    def test_and_selector(self):
        """Test AND operation."""
        lhs = IDSelectorRange(5, 15)
        rhs = IDSelectorRange(10, 20)
        selector = IDSelectorAnd(lhs, rhs)
        
        self.assertFalse(selector.is_member(4))   # Neither
        self.assertFalse(selector.is_member(7))   # Only lhs
        self.assertFalse(selector.is_member(17))  # Only rhs
        self.assertTrue(selector.is_member(12))   # Both
        
    def test_or_selector(self):
        """Test OR operation."""
        lhs = IDSelectorRange(5, 15)
        rhs = IDSelectorRange(10, 20)
        selector = IDSelectorOr(lhs, rhs)
        
        self.assertFalse(selector.is_member(4))   # Neither
        self.assertTrue(selector.is_member(7))    # Only lhs
        self.assertTrue(selector.is_member(17))   # Only rhs
        self.assertTrue(selector.is_member(12))   # Both
        
    def test_xor_selector(self):
        """Test XOR operation."""
        lhs = IDSelectorRange(5, 15)
        rhs = IDSelectorRange(10, 20)
        selector = IDSelectorXor(lhs, rhs)
        
        self.assertFalse(selector.is_member(4))   # Neither
        self.assertTrue(selector.is_member(7))    # Only lhs
        self.assertTrue(selector.is_member(17))   # Only rhs
        self.assertFalse(selector.is_member(12))  # Both
        
    def test_complex_operations(self):
        """Test complex combinations of selectors."""
        # Create (A AND B) OR (NOT C)
        a = IDSelectorRange(5, 15)
        b = IDSelectorRange(10, 20)
        c = IDSelectorArray([7, 12, 17])
        
        ab = IDSelectorAnd(a, b)
        not_c = IDSelectorNot(c)
        selector = IDSelectorOr(ab, not_c)
        
        # Test various cases
        self.assertTrue(selector.is_member(4))    # NOT C
        self.assertFalse(selector.is_member(7))   # In C
        self.assertTrue(selector.is_member(12))   # A AND B (despite being in C)
        self.assertTrue(selector.is_member(14))   # A AND B
        self.assertFalse(selector.is_member(17))  # In C
        self.assertTrue(selector.is_member(25))   # NOT C

if __name__ == '__main__':
    unittest.main()
