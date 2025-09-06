"""
Central feature flags for gating experimental/research kernels in production.

All flags read environment variables of the form METALFAISS_ENABLE_* and
return booleans. Defaults are conservative (disabled) so production code stays
stable unless explicitly enabled.
"""

from __future__ import annotations
import os


def _env_enabled(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def kernels_enabled(default: bool = False) -> bool:
    """Global switch for enabling custom Metal kernels.

    Default False keeps production on MLX ops unless explicitly turned on.
    """
    return _env_enabled("METALFAISS_ENABLE_KERNELS", default)


def ivf_fused_enabled(default: bool = False) -> bool:
    """Enable fused IVF scan+select kernels.

    Controlled by METALFAISS_ENABLE_IVF_FUSED; falls back to global switch if set.
    """
    if _env_enabled("METALFAISS_ENABLE_IVF_FUSED", None) is not None:
        return _env_enabled("METALFAISS_ENABLE_IVF_FUSED", default)
    return kernels_enabled(default)


def qr_kernels_enabled(default: bool = False) -> bool:
    """Enable QR projection/update kernels (in addition to MLX path)."""
    if _env_enabled("METALFAISS_ENABLE_QR_KERNELS", None) is not None:
        return _env_enabled("METALFAISS_ENABLE_QR_KERNELS", default)
    return kernels_enabled(default)


def svd_kernels_enabled(default: bool = False) -> bool:
    """Enable SVD Z-step kernels (in addition to MLX path)."""
    if _env_enabled("METALFAISS_ENABLE_SVD_KERNELS", None) is not None:
        return _env_enabled("METALFAISS_ENABLE_SVD_KERNELS", default)
    return kernels_enabled(default)

