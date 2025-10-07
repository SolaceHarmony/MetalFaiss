# Emberlint - MetalFaiss Zero-CPU Policy Enforcer

## Purpose
Enforce MetalFaiss's **ZERO-CPU policy** - all computation must stay on Apple Silicon GPU/Metal.

This tool catches:
- ❌ NumPy imports and usage (use MLX instead)
- ❌ CPU transfers: `.tolist()`, `.item()`, `.numpy()` calls
- ❌ Python operators on MLX arrays (use `mx.add()`, `mx.multiply()`, etc.)
- ❌ Python comparisons on MLX (use `mx.equal()`, `mx.less()`, etc.)
- ❌ Bitwise operations (use `mx.bitwise_and()`, etc.)
- ⚠️  Precision-reducing casts without boundary markers
- ⚠️  Tensor conversions between backends

**NO FLAGS NEEDED** - All checks are always enabled.

---

## Quick Start

### Basic Usage
```bash
# Scan directory
python emberlint.py python/metalfaiss

# Verbose output with details
python emberlint.py python/metalfaiss -v

# Summary only
python emberlint.py python/metalfaiss --summary

# JSON output for CI
python emberlint.py python/metalfaiss --json
```

### Common Commands
```bash
# Scan production code (exclude tests)
python emberlint.py python/metalfaiss --exclude unittest tests

# CI mode (always exit 0, just report)
python emberlint.py python/metalfaiss --exit-zero

# Full verbose scan
python emberlint.py python/metalfaiss -v
```

---

## Exit Codes
- **0** - No violations found (or `--exit-zero` flag used)
- **1** - Zero-CPU policy violations detected

---

## Exemptions

Mark specific lines with comments to exempt them:
```python
# For legitimate boundary conversions (test assertions, final output)
result = mx_array.tolist()  # boundary-ok

# Alternative marker
debug_val = mx_array.item()  # lint: allow-host-pull
```

**Use sparingly!** Only for:
- Test assertions
- Final output formatting for users
- Debug/logging (temporary)
- File I/O boundaries

---

## Configuration File (Optional)

Create `.emberlint.json` or `.emberlint.yml` at repo root:

```json
{
  "exclude": ["python/metalfaiss/unittest", "build", "docs"],
  "verbose": true
}
```

Supported keys:
- `exclude`: List of paths to skip
- `verbose`: Enable verbose output by default

---

## Policy Enforcement

### Always Enforced
1. **No NumPy** - Use MLX for all array operations
2. **No CPU Transfers** - Keep data on GPU (.tolist(), .item(), .numpy() forbidden)
3. **No Python Operators** - Use mx.add() not +, mx.multiply() not *, etc.
4. **No Python Comparisons** - Use mx.equal() not ==, mx.less() not <, etc.
5. **No Bitwise Python** - Use mx.bitwise_and() not &, mx.bitwise_or() not |, etc.

---

## See Also
- `ZERO_CPU_AUDIT.md` - Detailed audit report with patterns
- `docs/mlx/No-CPU-Math-Contract.md` - Policy explanation
- `docs/mlx/ARRAYS.md` - MLX array operations guide

