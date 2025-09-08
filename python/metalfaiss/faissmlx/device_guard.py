"""
Device guard helpers — enforce GPU (Metal) usage for compute-heavy paths.

Behavior
- By default, raises a RuntimeError if MLX Metal GPU is not available or the
  current default device is not GPU.
- Can be relaxed for CI/tests by setting METALFAISS_ALLOW_CPU=1 (not
  recommended for performance/correctness of this project).

Rationale
- Many kernels and MLX code paths here are designed for GPU execution. Running
  on CPU leads to severe slowdowns and unsupported fallbacks. We fail fast
  with a clear message to guide users to `mx.set_default_device(mx.gpu)`.
"""

from __future__ import annotations
import os
import mlx.core as mx


def _is_gpu_default_device() -> bool:
    try:
        # If MLX exposes metal and it is available, prefer that signal
        import mlx.core.metal as metal  # type: ignore
        if hasattr(metal, "is_available") and metal.is_available():
            return True
    except Exception:
        pass
    # Fallback: inspect default device string
    try:
        dev = str(mx.default_device()).lower()
        return ("gpu" in dev) or ("metal" in dev)
    except Exception:
        return False


def require_gpu(context: str = "") -> None:
    """Fail fast if running on CPU, unless explicitly allowed via env.

    Args:
        context: Optional string describing the call site for better error messages.
    """
    if os.environ.get("METALFAISS_ALLOW_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if not _is_gpu_default_device():
        where = f" for {context}" if context else ""
        raise RuntimeError(
            "MetalFaiss requires MLX Metal GPU for this operation"
            f"{where}. Set `mx.set_default_device(mx.gpu)` before calling, or set "
            "METALFAISS_ALLOW_CPU=1 to bypass at your own risk."
        )

