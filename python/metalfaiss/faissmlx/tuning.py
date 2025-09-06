"""
Static hardware parameter loading for kernels and heuristics.

This module reads a packaged JSON (`faissmlx/config/hardware_params.json`)
and exposes small helpers to pick tile sizes, band sizes, and QR dot mode
without runtime autotuning.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import json
import os
import importlib.resources as pkg_resources


def _load_params() -> Dict:
    try:
        with pkg_resources.files(__package__ + ".config").joinpath("hardware_params.json").open("r") as f:
            return json.load(f)
    except Exception:
        return {}


_PARAMS = _load_params()


def backend() -> str:
    try:
        import mlx.core.metal as _metal  # noqa: F401
        return "metal"
    except Exception:
        try:
            import mlx.core.cuda as _cuda  # noqa: F401
            return "cuda"
        except Exception:
            return "cpu"


def device_key() -> Tuple[str, str]:
    b = backend()
    if b == "metal":
        try:
            import mlx.core.metal as metal
            info = metal.device_info()
            name = str(info.get("device_name", ""))
            return b, name
        except Exception:
            return b, "default"
    if b == "cuda":
        try:
            import mlx.core.cuda as cuda
            info = cuda.device_info()  # type: ignore[attr-defined]
            # Try to pick a stable key (compute capability or name)
            cc = str(info.get("compute_capability", ""))
            name = str(info.get("name", ""))
            key = (cc or name or "default").lower()
            # Normalize common CC to sm_XX
            if cc:
                key = f"sm_{cc}"
            return b, key
        except Exception:
            return b, "default"
    return b, "default"


def get(section: str, key: str, default=None):
    b, dev = device_key()
    tree = _PARAMS.get(b, {})
    # First try exact device match
    if dev in tree and key in tree[dev]:
        return tree[dev][key]
    # Try partial name match (e.g., "Apple M3" from "Apple M3 Ultra")
    for k, v in tree.items():
        if k == "default":
            continue
        if k and k in dev and key in v:
            return v[key]
    # Fallback to backend default
    if "default" in tree and key in tree["default"]:
        return tree["default"][key]
    return default


def tiles_for_gemm() -> Tuple[Optional[str], Optional[str]]:
    """Return (av, atb) tile strings or (None, None) if not configured."""
    av = get("metal", "av") or get("cuda", "av")
    atb = get("metal", "atb") or get("cuda", "atb")
    return av, atb


def qr_dot_mode() -> str:
    return get("metal", "qr_dot_mode", default="auto")


def svd_band_and_streams() -> Tuple[Optional[int], Optional[int]]:
    band = get("metal", "svd_band") or get("cuda", "svd_band")
    streams = get("metal", "svd_streams") or get("cuda", "svd_streams")
    return band, streams

