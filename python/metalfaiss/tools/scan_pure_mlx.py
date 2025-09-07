"""
scan_pure_mlx.py — quick scanner for violations of the pure‑MLX contract.

Flags patterns:
 - Python operators on arrays: " ** ", " / ", " * " (heuristic: skip comments)
 - Host pulls: ".item(", ".tolist(", ".numpy("
 - float(…) / int(…) casts in code regions likely to be compute

Usage
  PYTHONPATH=python python -m python.metalfaiss.tools.scan_pure_mlx
"""

from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / 'python' / 'metalfaiss'

PATTERNS = {
    'host_pull': re.compile(r"\.(item|tolist|numpy)\("),
    'py_cast': re.compile(r"\b(float|int)\("),
    'pow': re.compile(r"\*\*\s*\d"),
    'div': re.compile(r"[^/]/[^/]")
}

def main():
    for path in SRC.rglob('*.py'):
        if 'unittest' in path.parts:
            continue
        # Skip demo helpers by default
        if path.name in {'demo_utils.py', 'setup.py'}:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        lines = text.splitlines()
        in_doc = False
        for i, line in enumerate(lines, 1):
            # crude docstring toggle
            if '"""' in line or "'''" in line:
                in_doc = not in_doc
            if in_doc:
                continue
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            for key, rx in PATTERNS.items():
                if rx.search(line):
                    print(f"{path}:{i}:{key}: {s}")

if __name__ == '__main__':
    main()
