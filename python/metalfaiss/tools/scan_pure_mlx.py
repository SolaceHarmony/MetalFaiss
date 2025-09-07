"""
scan_pure_mlx.py — quick scanner for violations of the pure‑MLX contract.

It reports likely uses of Python arithmetic with MLX arrays and host pulls.
This is a heuristic plus light AST inspection:
 - Python operators on arrays: +, -, *, /, **, unary - with lines that mention `mx.`
 - Host pulls: .item(, .tolist(, .numpy(
 - Python float()/int() casts

Usage
  PYTHONPATH=python python -m python.metalfaiss.tools.scan_pure_mlx
"""

from __future__ import annotations
import re
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / 'python' / 'metalfaiss'

PATTERNS = {
    'host_pull': re.compile(r"\.(item|tolist|numpy)\("),
    'py_cast': re.compile(r"\b(float|int)\("),
    'pow': re.compile(r"\*\*\s*\d"),
    'div': re.compile(r"[^/]/[^/]"),
}

OPS = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.Pow: '**',
    ast.MatMult: '@',
}

def _line_uses_mx(line: str) -> bool:
    return 'mx.' in line

def _has_mx(node: ast.AST) -> bool:
    found = False
    class W(ast.NodeVisitor):
        def visit_Attribute(self, n):
            nonlocal found
            if isinstance(n.value, ast.Name) and n.value.id == 'mx':
                found = True
            self.generic_visit(n)
        def visit_Call(self, n):
            self.visit(n.func)
            for a in n.args:
                self.visit(a)
            for k in n.keywords:
                self.visit(k.value)
    W().visit(node)
    return found

def _scan_ast_for_ops(path: Path, lines: list[str]):
    try:
        tree = ast.parse('\n'.join(lines))
    except Exception:
        return []
    findings = []

    class V(ast.NodeVisitor):
        def visit_BinOp(self, node: ast.BinOp):
            op = OPS.get(type(node.op))
            if op:
                line = lines[node.lineno - 1]
                # Skip string concatenations etc. if either side is a string literal
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    return
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                    return
                # Only flag when one of the operands syntactically involves mx.*
                if _has_mx(node.left) or _has_mx(node.right):
                    if op in {'&', '|', '^'}:
                        sugg = {
                            '&': 'mx.bitwise_and(a, b)',
                            '|': 'mx.bitwise_or(a, b)',
                            '^': 'mx.bitwise_xor(a, b)'
                        }[op]
                        findings.append((node.lineno, f'py_bitop:{op}', f"{line.strip()}    # suggest: {sugg}"))
                    else:
                        findings.append((node.lineno, f'py_op:{op}', line.strip()))
            self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                line = lines[node.lineno - 1]
                # Skip numeric literals like reshape(-1, ...)
                if isinstance(node.operand, ast.Constant):
                    return
                if _line_uses_mx(line):
                    findings.append((node.lineno, 'py_op:neg', line.strip()))
            if isinstance(node.op, ast.Invert):
                if _has_mx(node.operand):
                    line = lines[node.lineno - 1]
                    findings.append((node.lineno, 'py_bitop:~', f"{line.strip()}    # suggest: mx.bitwise_not(x)"))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            # Flag float(...) or int(...) when arguments involve mx.* expressions
            callee = node.func
            name = None
            if isinstance(callee, ast.Name):
                name = callee.id
            if name in {'float', 'int'}:
                # If any argument subtree contains mx.*, flag as mx cast
                for arg in node.args:
                    if _has_mx(arg):
                        line = lines[node.lineno - 1]
                        if name == 'float':
                            hint = 'suggest: on-device: expr.astype(mx.float32); boundary: float(expr.astype(mx.float32))'
                        else:
                            hint = 'suggest: on-device: expr.astype(mx.int32); boundary (index): int(expr.astype(mx.int32))'
                        findings.append((node.lineno, f'py_cast_mx:{name}', f"{line.strip()}    # {hint}"))
                        break
            if name == 'bool':
                for arg in node.args:
                    if _has_mx(arg):
                        line = lines[node.lineno - 1]
                        findings.append((node.lineno, 'py_cast_mx:bool', f"{line.strip()}    # suggest: use mx.any(expr) or mx.all(expr), not bool(expr)"))
                        break
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare):
            # Flag Python comparisons on mx.* expressions; suggest MLX patterns
            # Supports chained comparisons (a < b < c). We flag each operator.
            ops_map = {
                ast.Eq: ("==", "mx.equal(a, b)  (Alt: mx.equal(a, mx.array(0, dtype=a.dtype)))"),
                ast.NotEq: ("!=", "mx.not_equal(a, b)"),
                ast.Lt: ("<", "mx.less(a, b)"),
                ast.LtE: ("<=", "mx.less_equal(a, b)"),
                ast.Gt: (">", "mx.greater(a, b)"),
                ast.GtE: (">=", "mx.greater_equal(a, b)"),
            }
            # Only flag if any side syntactically involves mx.*
            has_mx_left = _has_mx(node.left)
            has_mx_any = has_mx_left or any(_has_mx(c) for c in node.comparators)
            if has_mx_any:
                line = lines[node.lineno - 1].strip()
                for op in node.ops:
                    sym, sugg = ops_map.get(type(op), (None, None))
                    if sym is None:
                        continue
                    findings.append((node.lineno, f'py_cmp:{sym}', f"{line}    # suggest: {sugg}"))
            self.generic_visit(node)

    V().visit(tree)
    return findings

def main():
    for path in SRC.rglob('*.py'):
        if 'unittest' in path.parts:
            continue
        # Skip demo/experiments/helpers by default
        if path.name in {'demo_utils.py', 'setup.py'}:
            continue
        if 'experiments' in path.parts:
            continue
        # Temporary suppression while HNSW is being refactored to array-native
        if any(p in path.as_posix() for p in [
            '/index/hnsw.py', '/index/hnsw_index.py', '/index/binary_hnsw_index.py']):
            pass  # do not early-continue; still scan for non-cast ops
        text = path.read_text(encoding='utf-8', errors='ignore')
        lines = text.splitlines()
        in_doc = False
        # AST-based operator scan (with mx heuristic)
        ast_findings = _scan_ast_for_ops(path, lines)
        cast_mx_lines = {ln for (ln, tag, _s) in ast_findings if tag.startswith('py_cast_mx')}
        for ln, tag, s in ast_findings:
            print(f"{path}:{ln}:{tag}: {s}")

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
                if key == 'py_cast' and i in cast_mx_lines:
                    # Already flagged by AST as py_cast_mx (more specific)
                    continue
                if rx.search(line):
                    print(f"{path}:{i}:{key}: {s}")

            # Device hints: discourage explicit CPU streams in hot paths
            if 'stream=' in line and 'mx.cpu' in line:
                print(f"{path}:{i}:cpu_stream: {s}    # suggest: prefer Metal stream or default device; see docs/mlx/using_streams.md")
            if 'Device.CPU' in line:
                print(f"{path}:{i}:device_cpu: {s}    # suggest: avoid forcing CPU; rely on default Metal device when available")
            if 'mx.cpu' in line and 'stream=' not in line:
                print(f"{path}:{i}:cpu_ref: {s}    # suggest: avoid explicit mx.cpu for compute; see docs/mlx/devices_and_streams.md")

if __name__ == '__main__':
    main()
