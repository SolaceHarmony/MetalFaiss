"""
Round-trip tests for index_factory and reverse_factory.
"""

import unittest
from metalfaiss import index_factory, reverse_factory
from metalfaiss.index.ivf_flat_index import IVFFlatIndex
from metalfaiss.index.hnsw_index import HNSWFlatIndex
from metalfaiss.index.flat_index import FlatIndex
from metalfaiss.index.product_quantizer_index import ProductQuantizerIndex
from metalfaiss.index.scalar_quantizer_index import ScalarQuantizerIndex
from metalfaiss.index.pre_transform_index import PreTransformIndex
from metalfaiss.index.id_map import IDMap


class TestFactoryRoundTrip(unittest.TestCase):
    def test_basic_roundtrip(self):
        keys = [
            "Flat",
            "IVF100,Flat",
            "HNSW16",
            "PQ8",
            "SQ8",
            "IDMap,Flat",
            "PCA32,Flat",
            "PCAR32,Flat",
            "PCAW32,Flat",
            "PCAWR16,Flat",
            "OPQ16_64,Flat",
            "ITQ32,Flat",
            "RR64,Flat",
            "Pad128,Flat",
            "L2norm,Flat",
            "RFlat",
        ]
        d = 64
        for key in keys:
            with self.subTest(key=key):
                idx = index_factory(d, key)
                # reverse may normalize, so build from reverse and compare types
                rev = reverse_factory(idx)
                idx2 = index_factory(d, rev)
                self.assertIsInstance(idx2, type(idx) if not isinstance(idx, PreTransformIndex) else PreTransformIndex)

    def test_ivf_attrs(self):
        idx = index_factory(64, "IVF200,Flat")
        self.assertIsInstance(idx, IVFFlatIndex)
        self.assertEqual(idx.nlist, 200)


if __name__ == '__main__':
    unittest.main()
