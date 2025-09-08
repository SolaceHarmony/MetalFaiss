# (Removed shebang - utility is importable; run with the project's Python interpreter)
"""
mlxtop.py - Lightweight MLX device/streams/memory monitor

Inspired by xLSTM's xltop.py, this tool reports:
- Device info (name, arch, memory sizes)
- Active/cache/peak memory (MLX tracked)
- Default device/stream
- Relevant MetalFaiss env toggles (SVD/QR kernel, compile)
- Optional Ray summary if ray is installed and a local head is running

Usage:
  python -m metalfaiss.tools.mlxtop [--interval 0.5] [--once]
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime

try:
    import mlx.core as mx
    from mlx.core import metal as mxmetal
except Exception as e:
    print("ERROR: MLX not available:", e, file=sys.stderr)
    sys.exit(1)


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024.0:
            return f"{s:.1f}{u}"
        s /= 1024.0
    return f"{s:.1f}PB"


def _env_flags() -> dict:
    keys = [
        "METALFAISS_USE_SVD_KERNEL",
        "METALFAISS_USE_QR_KERNEL",
        "METALFAISS_USE_COMPILE",
        "METALFAISS_FORCE_SVD",
        "METALFAISS_FORCE_QR",
    ]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def _ray_summary() -> str | None:
    try:
        import ray
        try:
            info = ray._private.worker.global_worker
            if info and info.mode != ray._private.worker.LOCAL_MODE:
                return "ray: head running (multi-process)"
            else:
                return "ray: local_mode or not initialized"
        except Exception:
            return "ray: installed but not initialized"
    except Exception:
        return None


def print_header():
    di = mxmetal.device_info()
    print("MLX Device Info")
    print(f"- name: {di.get('device_name')}")
    print(f"- arch: {di.get('architecture')}")
    print(f"- memory_size: {_fmt_bytes(int(di.get('memory_size', 0)))}")
    print(f"- max_recommended_working_set_size: {_fmt_bytes(int(di.get('max_recommended_working_set_size', 0)))}")
    print(f"- max_buffer_length: {_fmt_bytes(int(di.get('max_buffer_length', 0)))}")
    print(f"Default device: {mx.default_device()}")
    rf = _ray_summary()
    if rf:
        print(f"{rf}")
    env = _env_flags()
    if env:
        print("Env flags:")
        for k, v in env.items():
            print(f"  - {k}={v}")
    print()


def print_row():
    active = mx.get_active_memory()
    cache = mx.get_cache_memory()
    peak = mx.get_peak_memory()
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] active={_fmt_bytes(active)} | cache={_fmt_bytes(cache)} | peak={_fmt_bytes(peak)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.5, help="refresh interval (seconds)")
    ap.add_argument("--once", action="store_true", help="print a single line and exit")
    args = ap.parse_args(argv)

    print_header()
    if args.once:
        print_row(); return 0
    try:
        while True:
            print_row()
            sys.stdout.flush()
            time.sleep(max(0.05, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

