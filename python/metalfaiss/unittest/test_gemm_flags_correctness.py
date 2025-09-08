"""
Correctness checks for GEMM kernels across feature flags.

Verifies B = A@V and Z = A^T@B against mx.matmul for a few awkward shapes
and toggle combinations (DB, V4, PAD_ATB) with square tiles.
"""

import os
import unittest
import mlx.core as mx

from ..faissmlx.kernels import gemm_kernels as gk


def _max_abs_err(x: mx.array, y: mx.array) -> float:
    d = mx.abs(x - y)
    return float(mx.max(d).item())  # boundary-ok


class TestGemmFlagsCorrectness(unittest.TestCase):
    def _run_case(self, m, n, k, db, v4, pad, sq):
        os.environ["METALFAISS_USE_GEMM_KERNEL"] = "1"
        os.environ["METALFAISS_GEMM_DB"] = "1" if db else "0"
        os.environ["METALFAISS_GEMM_V4"] = "1" if v4 else "0"
        os.environ["METALFAISS_GEMM_PAD_ATB"] = "1" if pad else "0"
        os.environ["METALFAISS_GEMM_TILE_SQ"] = str(sq)
        gk.reset_gemm_kernels()

        A = mx.random.normal(shape=(m, n)).astype(mx.float32)
        V = mx.random.normal(shape=(n, k)).astype(mx.float32)
        B = gk.gemm_av(A, V)
        Z = gk.gemm_at_b(A, B)
        B_ref = mx.matmul(A, V)
        Z_ref = mx.matmul(mx.transpose(A), B_ref)
        mx.eval(B, Z, B_ref, Z_ref)
        berr = _max_abs_err(B, B_ref)
        zerr = _max_abs_err(Z, Z_ref)
        self.assertLess(berr, 5e-4, f"B err too large: {berr}")
        self.assertLess(zerr, 5e-3, f"Z err too large: {zerr}")

    def test_flags_small_shapes(self):
        shapes = [(33, 29, 31), (64, 64, 64)]
        toggles = [
            (False, False, False, 16),
            (True,  False, False, 16),
            (False, True,  False, 16),
            (False, False, True,  16),
            (True,  True,  False, 16),
            (True,  False, True,  16),
        ]
        for (m, n, k) in shapes:
            for (db, v4, pad, sq) in toggles:
                with self.subTest(shape=(m, n, k), db=db, v4=v4, pad=pad, sq=sq):
                    self._run_case(m, n, k, db, v4, pad, sq)


if __name__ == "__main__":
    unittest.main()
