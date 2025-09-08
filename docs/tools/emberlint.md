Emberlint (MetalFaiss Linter)

Purpose
- Enforce pure‑MLX, GPU‑first practices across the Python codebase.
- Catch NumPy imports/usages, host pulls, Python operators on MLX arrays, precision‑reducing casts, and GPU enforcement gaps.

Quick Start
- Full scan with human‑readable output:
  - `python emberlint.py python -v`
- Focus on one class of issues (single‑issue filtering):
  - `python emberlint.py python --operators-only -v`
  - `python emberlint.py python --mlx-only -v`  # host pulls, comparisons, bitwise
  - `python emberlint.py python --gpu-only -v`
- JSON output (for CI):
  - `python emberlint.py python --json > emberlint.json`
  - Summary only: `python emberlint.py python --json-summary`
- Per‑directory summary only:
  - `python emberlint.py python --summary-only`

Exit Codes
- Non‑zero when issues are found. Override with `--exit-zero` to always return 0 (useful for report‑only CI jobs).
- Use `--fail-on` to constrain what counts as a failure, e.g.:
  - `--fail-on operators mlx gpu` (ignore style/types, but fail on these categories)

Config File
- Place `.emberlint.json` or `.emberlint.yml` at repo root (or any ancestor of your scan path).
- Supported keys:
  - `exclude: ["python/metalfaiss/unittest", "build"]`
  - `fail_on: ["operators", "mlx", "gpu"]`
  - `verbose: true`
  - `summary_only: true`
  - `json: true` or `json_summary: true`

Examples
```
{
  "exclude": ["python/metalfaiss/unittest"],
  "fail_on": ["operators", "mlx", "gpu"],
  "summary_only": true
}
```

Notes
- Type checking uses mypy when available; otherwise it’s skipped.
- Style checks are currently disabled (placeholder for pycodestyle).
- MLX hints suggest device‑safe replacements and boundary casts when needed.

