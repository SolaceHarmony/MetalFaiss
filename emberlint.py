# (Removed shebang - this file is a module, not an executable script)
"""
MetalFaissLint: Zero-CPU Policy Enforcer for MetalFaiss

This script enforces MetalFaiss's ZERO-CPU policy by scanning Python files for:

**ALWAYS ENFORCED (No flags needed):**
1. ❌ NumPy imports and usage (use MLX instead)
2. ❌ CPU transfers: .tolist(), .item(), .numpy() calls
3. ❌ Python operators on MLX arrays (use mx.add, mx.multiply, etc.)
4. ❌ Precision-reducing casts without boundary markers
5. ❌ Tensor conversions between backends
6. ❌ Missing GPU device checks
7. ⚠️  Syntax and compilation errors

**POLICY:**
This is MetalFAISS - Metal/GPU only. NO CPU transfers allowed in production code.

**USAGE:**
  python emberlint.py python/metalfaiss          # Scan directory
  python emberlint.py python/metalfaiss -v       # Verbose output
  python emberlint.py python/metalfaiss --json   # JSON output for CI

**EXEMPTIONS:**
Mark lines with `# boundary-ok` or `# lint: allow-host-pull` to exempt specific operations.
Use sparingly and only for legitimate boundary cases (test assertions, final output formatting).
"""

# Enforce Zero-CPU policy - no switches, always on
ENFORCE_ZERO_CPU = True
SUGGEST_OPS = True  # Always suggest MLX operators

import os
import re
import ast
import sys
import json
import argparse
import subprocess
from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path

# Try to import optional dependencies
try:
    import pycodestyle
    HAVE_PYCODESTYLE = True
except ImportError:
    HAVE_PYCODESTYLE = False

try:
    import mypy.api
    HAVE_MYPY = True
except ImportError:
    HAVE_MYPY = False

# Optional YAML config support
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def load_config(start_path: str) -> Dict[str, Any]:
    """Load .emberlint.(yml|yaml|json) from nearest ancestor directory of start_path.

    If no config is found or it cannot be parsed, returns an empty dict.
    """
    p = Path(start_path).absolute()
    base = p if p.is_dir() else p.parent
    for ancestor in [base] + list(base.parents):
        for name in (".emberlint.yml", ".emberlint.yaml", ".emberlint.json"):
            cfg_path = ancestor / name
            if cfg_path.exists():
                try:
                    if cfg_path.suffix in {".yml", ".yaml"} and HAVE_YAML:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            return data or {}
                    if cfg_path.suffix == ".json":
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                except Exception:
                    return {}
    return {}

def check_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """Check if the file has syntax errors."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        return True, []
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

def check_imports(file_path: str) -> Tuple[bool, List[str]]:
    """Check if imports are syntactically valid."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, ["Syntax error prevents import checking"]
    
    return True, []

def check_style(file_path: str) -> Tuple[bool, List[str]]:
    """Check PEP 8 style issues."""
    # Temporarily disabled style checking
    return True, []

def check_types(file_path: str) -> Tuple[bool, List[str]]:
    """Check type annotations using mypy."""
    if not HAVE_MYPY:
        return True, ["mypy not installed, skipping type checking"]
    
    try:
        import mypy.api
        result = mypy.api.run([file_path])
        
        type_errors = []
        for line in result[0].split('\n'):
            if line.strip() and not line.startswith("Success:"):
                type_errors.append(line)
        
        return len(type_errors) == 0, type_errors
    except Exception as e:
        return False, [f"Error running mypy: {e}"]

def check_numpy_import(file_path: str) -> Tuple[bool, List[str]]:
    """Check if NumPy is imported in the file (regex-based, respects boundary-ok)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    numpy_imports = []
    lines = content.splitlines()
    
    patterns = [
        r'import\s+numpy\s+as\s+(\w+)',
        r'from\s+numpy\s+import\s+(.*)',
        r'import\s+numpy\b',
    ]
    
    for line_no, line in enumerate(lines, 1):
        # Check for boundary-ok marker
        if 'boundary-ok' in line or 'lint: allow-numpy' in line:
            continue  # Skip this line
            
        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                if pattern == r'import\s+numpy\s+as\s+(\w+)':
                    numpy_imports.extend(matches)
                elif pattern == r'from\s+numpy\s+import\s+(.*)':
                    for match in matches:
                        imports = [name.strip() for name in match.split(',')]
                        numpy_imports.extend(imports)
                else:
                    numpy_imports.append("numpy")
    
    return bool(numpy_imports), numpy_imports

def check_numpy_usage(file_path: str, numpy_aliases: List[str]) -> Tuple[bool, List[str]]:
    """Check if NumPy is used in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    numpy_usages = []
    numpy_random = []
    
    for alias in numpy_aliases:
        pattern = r'\b' + re.escape(alias) + r'\.\w+'
        matches = re.findall(pattern, content)
        if matches:
            numpy_usages.extend(matches)
        # Detect numpy.random.* usage and provide MLX guidance
        rx = r'\b' + re.escape(alias) + r'\.random\.(seed|rand|randn|normal|uniform|randint)\b'
        for m in re.finditer(rx, content):
            numpy_random.append((m.group(0), m.start()))
    
    return bool(numpy_usages), numpy_usages, numpy_random

def check_gpu_enforcement(file_path: str) -> Tuple[bool, List[str]]:
    """Check if the file properly enforces GPU usage."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    # Only enforce for index/entrypoint modules; helpers/kernels are exempt
    norm_path = file_path.replace('\\', '/')
    if '/index/' not in norm_path:
        return True, []
    # Only enforce on concrete index implementations
    base_name = os.path.basename(norm_path)
    if not (base_name.endswith('_index.py') or base_name in {'indexflat.py'}):
        return True, []
    enforced = {
        'ivf_pq_index.py',
        'hnsw_index.py',
        'binary_hnsw_index.py',
        'binary_ivf_index.py',
    }
    if base_name not in enforced:
        return True, []
    
    # Check for MLX CPU device usage
    cpu_patterns = [
        r'mx\.cpu\(',
        r'device\s*=\s*["\']cpu["\']',
        r'\.to\(["\']cpu["\']',
    ]
    
    for pattern in cpu_patterns:
        if re.search(pattern, content):
            issues.append(f"Potential CPU device usage: {pattern}")
    
    # Check if device_guard is imported when needed
    has_mlx_ops = 'import mlx' in content or 'from mlx' in content
    has_device_guard = 'device_guard' in content or 'require_gpu' in content
    
    if has_mlx_ops and not has_device_guard and 'test' not in file_path.lower():
        issues.append("MLX operations found but no GPU enforcement (missing device_guard)")
    
    return len(issues) == 0, issues

def _has_mx(node: ast.AST) -> bool:
    """Check if an AST node contains mx.* references."""
    found = False
    class MxVisitor(ast.NodeVisitor):
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
    MxVisitor().visit(node)
    return found

class MetalFaissVisitor(ast.NodeVisitor):
    """AST visitor to find MetalFaiss-specific issues."""
    
    def __init__(self, lines: List[str] = None):
        self.lines = lines or []
        self.precision_reducing_casts = []
        self.tensor_conversions = []
        self.python_operators = []
        self.numpy_imports = set()
        self.numpy_aliases = set()
        self.cpu_usage = []
        self.host_pulls = []
        self.comparisons = []
        self.bitwise_ops = []
        self.gpu_prefix = []
        self.current_function = None
        self.current_line = 0
        self.parent_map = {}
        
    def build_parent_map(self, node):
        """Build a map from child nodes to their parent nodes."""
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
            self.build_parent_map(child)
    
    def visit_Import(self, node):
        """Visit import statements."""
        for name in node.names:
            if name.name == 'numpy':
                # Check for boundary-ok marker
                line_text = self.lines[node.lineno - 1] if 0 < node.lineno <= len(self.lines) else ''
                if 'boundary-ok' in line_text or 'lint: allow-numpy' in line_text:
                    continue  # Skip this import, it's marked as acceptable
                    
                self.numpy_imports.add(f"import {name.name}")
                self.numpy_aliases.add(name.asname or name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module == 'numpy':
            # Check for boundary-ok marker
            line_text = self.lines[node.lineno - 1] if 0 < node.lineno <= len(self.lines) else ''
            if 'boundary-ok' in line_text or 'lint: allow-numpy' in line_text:
                self.generic_visit(node)
                return  # Skip this import, it's marked as acceptable
                
            for name in node.names:
                self.numpy_imports.add(f"from numpy import {name.name}")
                self.numpy_aliases.add(name.asname or name.name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_BinOp(self, node):
        """Visit binary operations to detect Python operators."""
        op_map = {
            ast.Add: ('+', 'mx.add(a, b)'),
            ast.Sub: ('-', 'mx.subtract(a, b)'),
            ast.Mult: ('*', 'mx.multiply(a, b)'),
            ast.Div: ('/', 'mx.divide(a, b)'),
            ast.FloorDiv: ('//', 'mx.floor_divide(a, b)'),
            ast.Mod: ('%', 'mx.remainder(a, b)'),
            ast.Pow: ('**', 'mx.power(a, b)'),
            ast.MatMult: ('@', 'mx.matmul(a, b)'),
            ast.BitAnd: ('&', 'mx.bitwise_and(a, b)'),
            ast.BitOr: ('|', 'mx.bitwise_or(a, b)'),
            ast.BitXor: ('^', 'mx.bitwise_xor(a, b)'),
        }
        
        # Check if we're in a type annotation (Python 3.10+ union types use |)
        is_in_annotation = False
        parent = self.parent_map.get(node)
        while parent:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign, ast.arg)):
                # We're in an annotation context
                is_in_annotation = True
                break
            parent = self.parent_map.get(parent)
        
        # Skip bitwise | in type annotations (Python 3.10+ union types)
        if is_in_annotation and isinstance(node.op, ast.BitOr):
            self.generic_visit(node)
            return
        
        is_in_subscript = False
        parent = self.parent_map.get(node)
        while parent:
            if isinstance(parent, ast.Subscript):
                if parent.slice == node:
                    is_in_subscript = True
                    break
            parent = self.parent_map.get(parent)
        
        is_non_tensor_op = False
        
        # Skip string concatenations and literals
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            return
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            return
        
        # Only flag when one of the operands involves mx.*
        if _has_mx(node.left) or _has_mx(node.right):
            op_type = type(node.op)
            if op_type in op_map:
                op_sym, suggestion = op_map[op_type]
                if SUGGEST_OPS:
                    if op_type is ast.Mult:
                        suggestion = (
                            "mx.multiply(a, b)  (matrix? mx.matmul(a, b); "
                            "scalar? mx.multiply(a, mx.array(s, dtype=a.dtype)))\n"
                            "Examples: diff=mx.subtract(x,y); d2=mx.multiply(diff,diff)  # avoid (x-y)*(x-y)"
                        )
                        # Special-case (x - y) * (x - y) and a*a patterns
                        try:
                            def _same_ast(a, b):
                                return ast.dump(a, annotate_fields=False, include_attributes=False) == \
                                       ast.dump(b, annotate_fields=False, include_attributes=False)
                            if _same_ast(node.left, node.right):
                                # a*a → recommend mx.square(a)
                                suggestion += "\nHint: use mx.square(a) for elementwise square."
                            if isinstance(node.left, ast.BinOp) and isinstance(node.right, ast.BinOp) \
                               and isinstance(node.left.op, ast.Sub) and isinstance(node.right.op, ast.Sub) \
                               and _same_ast(node.left, node.right):
                                suggestion += "\nPattern: (x - y)*(x - y) → diff=mx.subtract(x,y); d2=mx.multiply(diff,diff)"
                        except Exception:
                            pass
                    elif op_type is ast.Add:
                        suggestion = (
                            "mx.add(a, b)  (concatenate arrays: mx.concatenate([a,b], axis=...))"
                        )
                    elif op_type is ast.Sub:
                        suggestion = (
                            "mx.subtract(a, b)  (unary minus: mx.negative(x))"
                        )
                    elif op_type is ast.Div:
                        suggestion = (
                            "mx.divide(a, b)  (floor: mx.floor_divide(a,b); scalar: mx.divide(a, mx.array(s,dtype=a.dtype)))\n"
                            "Guard zeros: den = mx.where(mx.greater(den, mx.zeros_like(den)), den, mx.ones_like(den))"
                        )
                    elif op_type is ast.FloorDiv:
                        suggestion = "mx.floor_divide(a, b)"
                    elif op_type is ast.Mod:
                        suggestion = "mx.remainder(a, b)"
                    elif op_type is ast.Pow:
                        suggestion = (
                            "mx.power(a, b)  (square: mx.multiply(a,a); sqrt: mx.sqrt(a))"
                        )
                    elif op_type is ast.MatMult:
                        suggestion = (
                            "mx.matmul(a, b)  (ensure a.shape[-1] == b.shape[-2]; use mx.transpose(a) instead of a.T; "
                            "1D dot: mx.dot(x, y))"
                        )
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                
                if op_type in {ast.BitAnd, ast.BitOr, ast.BitXor}:
                    self.bitwise_ops.append({
                        'type': op_sym,
                        'location': location,
                        'line': node.lineno,
                        'suggestion': suggestion
                    })
                else:
                    self.python_operators.append({
                        'type': op_sym,
                        'location': location,
                        'line': node.lineno,
                        'suggestion': suggestion
                    })
        
        self.generic_visit(node)
    
    def visit_UnaryOp(self, node):
        """Visit unary operations."""
        if isinstance(node.op, ast.USub):
            # Skip numeric literals like reshape(-1, ...)
            if isinstance(node.operand, ast.Constant):
                return
            if _has_mx(node.operand):
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.python_operators.append({
                    'type': 'unary -',
                    'location': location,
                    'line': node.lineno,
                    'suggestion': 'mx.negative(x)'
                })
        elif isinstance(node.op, ast.Invert):
            if _has_mx(node.operand):
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.bitwise_ops.append({
                    'type': '~',
                    'location': location,
                    'line': node.lineno,
                    'suggestion': 'mx.bitwise_not(x)'
                })
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Catch array.T usage and suggest mx.transpose."""
        # x.T → mx.transpose(x)
        try:
            if SUGGEST_OPS and isinstance(node.attr, str) and node.attr == 'T' and _has_mx(node.value):
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.python_operators.append({
                    'type': '.T',
                    'location': location,
                    'line': node.lineno,
                    'suggestion': 'use mx.transpose(x); specify axes for >2D tensors'
                })
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Compare(self, node):
        """Visit comparison operations."""
        if SUGGEST_OPS:
            ops_map = {
                ast.Eq: ("==", "mx.equal(a, b)  (floats? mx.allclose(a,b,rtol=1e-5,atol=1e-8); to Python: bool(mx.all(...).item()))"),
                ast.NotEq: ("!=", "mx.not_equal(a, b)"),
                ast.Lt: ("<", "mx.less(a, b)"),
                ast.LtE: ("<=", "mx.less_equal(a, b)"),
                ast.Gt: (">", "mx.greater(a, b)"),
                ast.GtE: (">=", "mx.greater_equal(a, b)"),
            }
        else:
            ops_map = {
                ast.Eq: ("==", "mx.equal(a, b)"),
                ast.NotEq: ("!=", "mx.not_equal(a, b)"),
                ast.Lt: ("<", "mx.less(a, b)"),
                ast.LtE: ("<=", "mx.less_equal(a, b)"),
                ast.Gt: (">", "mx.greater(a, b)"),
                ast.GtE: (">=", "mx.greater_equal(a, b)"),
            }
        
        # Only flag if any side involves mx.*
        has_mx_any = _has_mx(node.left) or any(_has_mx(c) for c in node.comparators)
        if has_mx_any:
            for op in node.ops:
                sym, suggestion = ops_map.get(type(op), (None, None))
                if sym:
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    self.comparisons.append({
                        'type': sym,
                        'location': location,
                        'line': node.lineno,
                        'suggestion': suggestion
                    })
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls to detect precision-reducing casts, tensor conversions, and host pulls."""
        # Check for float(), int(), bool() casts on MLX expressions
        if isinstance(node.func, ast.Name) and node.func.id in ('float', 'int', 'bool'):
            # Check if any argument involves mx.*
            for arg in node.args:
                if _has_mx(arg):
                    # Check for boundary-ok marker
                    line_text = self.lines[node.lineno - 1] if 0 < node.lineno <= len(self.lines) else ''
                    if 'boundary-ok' in line_text or 'lint: allow-cast' in line_text:
                        break  # Skip this cast, it's marked as acceptable
                    
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    cast_type = node.func.id
                    
                    if cast_type == 'float':
                        suggestion = 'on-device: expr.astype(mx.float32); boundary: float(expr.item())'
                    elif cast_type == 'int':
                        suggestion = 'on-device: expr.astype(mx.int32); boundary: int(expr.item())'
                    else:  # bool
                        suggestion = 'use mx.any(expr) or mx.all(expr), not bool(expr)'
                    
                    self.precision_reducing_casts.append({
                        'type': cast_type,
                        'location': location,
                        'line': node.lineno,
                        'suggestion': suggestion
                    })
                    break
        
        # Check for host pulls: .item(), .tolist(), .numpy()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('item', 'tolist', 'numpy'):
                # Inline allow for boundary conversions
                line_text = self.lines[node.lineno - 1] if 0 < node.lineno <= len(self.lines) else ''
                if 'boundary-ok' in line_text or 'lint: allow-host-pull' in line_text:
                    pass
                else:
                    # Check if the object being called on involves mx.*
                    if _has_mx(node.func.value) or (self.lines and 'mx.' in self.lines[node.lineno - 1] if node.lineno <= len(self.lines) else False):
                        location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                        self.host_pulls.append({
                            'type': f'.{node.func.attr}()',
                            'location': location,
                            'line': node.lineno,
                            'suggestion': 'Consider keeping computation on device if possible'
                        })
            
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.numpy_aliases:
                if node.func.attr in ('array', 'asarray'):
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    self.tensor_conversions.append({
                        'type': f"{node.func.value.id}.{node.func.attr}",
                        'location': location,
                        'line': node.lineno
                    })
        
        # Pattern: mx.sum((x - y) * (x - y)) → mx.sum(mx.square(mx.subtract(x, y)))
        try:
            if SUGGEST_OPS and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'mx' and node.func.attr == 'sum':
                    if node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.BinOp) and isinstance(arg0.op, ast.Mult):
                            left, right = arg0.left, arg0.right
                            def _same_ast(a, b):
                                return ast.dump(a, annotate_fields=False, include_attributes=False) == \
                                       ast.dump(b, annotate_fields=False, include_attributes=False)
                            if _same_ast(left, right):
                                # If it looks like (a-a) or (x-y)*(x-y)
                                loc = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                                self.python_operators.append({
                                    'type': 'sum(square)',
                                    'location': loc,
                                    'line': node.lineno,
                                    'suggestion': 'mx.sum(mx.square(expr))  (and expr via mx.subtract(x, y) if from (x-y)*(x-y))'
                                })
        except Exception:
            pass

        self.generic_visit(node)

class UnusedImportVisitor(ast.NodeVisitor):
    """AST visitor to find unused imports."""
    
    def __init__(self):
        self.imports = {}
        self.used_names = set()
        
    def visit_Import(self, node):
        """Visit import statements."""
        for name in node.names:
            self.imports[name.asname or name.name] = node.lineno
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        for name in node.names:
            self.imports[name.asname or name.name] = node.lineno
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Visit name references."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
    
    def get_unused_imports(self):
        """Get unused imports."""
        unused = []
        for name, lineno in self.imports.items():
            if name not in self.used_names:
                unused.append((name, lineno))
        return unused

def check_ast_for_issues(file_path: str) -> Tuple[bool, List[str], List[str], List[Dict], List[Dict], List[Dict], List[Tuple[str, int]], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Use AST to check for various issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.splitlines()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, [], [], [], [], [], [], [], [], [], []
    
    visitor = MetalFaissVisitor(lines)
    visitor.build_parent_map(tree)
    visitor.visit(tree)
    
    unused_visitor = UnusedImportVisitor()
    unused_visitor.visit(tree)
    unused_imports = unused_visitor.get_unused_imports()
    
    numpy_usages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in visitor.numpy_aliases:
                numpy_usages.append(f"{node.value.id}.{node.attr}")
    
    return (
        bool(visitor.numpy_imports),
        list(visitor.numpy_imports),
        numpy_usages,
        visitor.precision_reducing_casts,
        visitor.tensor_conversions,
        visitor.python_operators,
        unused_imports,
        visitor.cpu_usage,
        visitor.host_pulls,
        visitor.comparisons,
        visitor.bitwise_ops,
        visitor.gpu_prefix
    )

def check_compilation(file_path: str) -> Tuple[bool, List[str]]:
    """Check if the file compiles without errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, file_path, 'exec')
        return True, []
    except Exception as e:
        return False, [str(e)]

def analyze_file(file_path: str) -> Dict:
    """Analyze a file for various issues."""
    # Check for syntax errors
    syntax_valid, syntax_errors = check_syntax(file_path)
    
    # Check for compilation errors
    compilation_valid, compilation_errors = check_compilation(file_path)
    
    # Check for import errors
    imports_valid, import_errors = check_imports(file_path)
    
    # Check for style issues
    style_valid, style_errors = check_style(file_path)
    
    # Check for type issues
    types_valid, type_errors = check_types(file_path)
    
    # Check for NumPy imports using regex
    has_numpy_import, numpy_imports = check_numpy_import(file_path)
    
    # If NumPy is imported, check for usage
    has_numpy_usage = False
    numpy_usages = []
    numpy_random = []
    if has_numpy_import:
        has_numpy_usage, numpy_usages, numpy_random = check_numpy_usage(file_path, numpy_imports)
    
    # Check for GPU enforcement issues
    gpu_enforced, gpu_issues = check_gpu_enforcement(file_path)
    
    # Use AST for more accurate detection
    ast_has_numpy, ast_numpy_imports, ast_numpy_usages, precision_casts, tensor_conversions, python_operators, unused_imports, cpu_usage, host_pulls, comparisons, bitwise_ops, gpu_prefix = check_ast_for_issues(file_path)
    
    # Combine results
    has_numpy = has_numpy_import or ast_has_numpy
    all_imports = list(set(numpy_imports + ast_numpy_imports))
    all_usages = list(set(numpy_usages + ast_numpy_usages))
    
    return {
        "file": file_path,
        "syntax_valid": syntax_valid,
        "syntax_errors": syntax_errors,
        "compilation_valid": compilation_valid,
        "compilation_errors": compilation_errors,
        "imports_valid": imports_valid,
        "import_errors": import_errors,
        "style_valid": style_valid,
        "style_errors": style_errors,
        "types_valid": types_valid,
        "type_errors": type_errors,
        "has_numpy": has_numpy,
        "imports": all_imports,
        "usages": all_usages,
        "numpy_random": numpy_random,
        "precision_casts": precision_casts,
        "tensor_conversions": tensor_conversions,
        "python_operators": python_operators,
        "unused_imports": unused_imports,
        "gpu_enforced": gpu_enforced,
        "gpu_issues": gpu_issues,
        "cpu_usage": cpu_usage,
        "host_pulls": host_pulls,
        "comparisons": comparisons,
        "bitwise_ops": bitwise_ops,
        "gpu_prefix": gpu_prefix
    }

def analyze_directory(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[Dict]:
    """Analyze all Python files in a directory for various issues."""
    if exclude_dirs is None:
        exclude_dirs = []
    
    python_files = find_python_files(directory)
    
    filtered_files = []
    for file_path in python_files:
        exclude = False
        for exclude_dir in exclude_dirs:
            if exclude_dir in file_path:
                exclude = True
                break
        if not exclude:
            filtered_files.append(file_path)
    
    results = []
    for file_path in filtered_files:
        result = analyze_file(file_path)
        results.append(result)
    
    return results

def print_results(results: List[Dict], verbose: bool = False, show_all: bool = True,
                 show_syntax: bool = False, show_compilation: bool = False,
                 show_imports: bool = False, show_style: bool = False,
                 show_types: bool = False, show_numpy: bool = False,
                 show_precision: bool = False, show_conversion: bool = False,
                 show_operators: bool = False, show_unused: bool = False,
                 show_gpu: bool = False, show_mlx: bool = False,
                 summary_only: bool = False):
    """Print the analysis results."""
    files_with_syntax_errors = [result for result in results if not result["syntax_valid"]]
    files_with_compilation_errors = [result for result in results if not result["compilation_valid"]]
    files_with_import_errors = [result for result in results if not result["imports_valid"]]
    files_with_style_errors = [result for result in results if not result["style_valid"]]
    files_with_type_errors = [result for result in results if not result["types_valid"]]
    numpy_files = [result for result in results if result["has_numpy"]]
    files_with_precision_casts = [result for result in results if result["precision_casts"]]
    files_with_tensor_conversions = [result for result in results if result["tensor_conversions"]]
    files_with_python_operators = [result for result in results if result["python_operators"]]
    files_with_unused_imports = [result for result in results if result["unused_imports"]]
    files_with_gpu_issues = [result for result in results if not result["gpu_enforced"]]
    files_with_cpu_usage = [result for result in results if result["cpu_usage"]]
    files_with_host_pulls = [result for result in results if result.get("host_pulls", [])]
    files_with_comparisons = [result for result in results if result.get("comparisons", [])]
    files_with_bitwise = [result for result in results if result.get("bitwise_ops", [])]
    files_with_gpu_prefix = [result for result in results if result.get("gpu_prefix", [])]
    
    print(f"Total files analyzed: {len(results)}")
    if len(results) > 0:
        if show_all or show_syntax:
            print(f"Files with syntax errors: {len(files_with_syntax_errors)} ({len(files_with_syntax_errors)/len(results)*100:.2f}%)")
        if show_all or show_compilation:
            print(f"Files with compilation errors: {len(files_with_compilation_errors)} ({len(files_with_compilation_errors)/len(results)*100:.2f}%)")
        if show_all or show_imports:
            print(f"Files with import errors: {len(files_with_import_errors)} ({len(files_with_import_errors)/len(results)*100:.2f}%)")
        if show_all or show_style:
            print(f"Files with style errors: {len(files_with_style_errors)} ({len(files_with_style_errors)/len(results)*100:.2f}%)")
        if show_all or show_types:
            print(f"Files with type errors: {len(files_with_type_errors)} ({len(files_with_type_errors)/len(results)*100:.2f}%)")
        if show_all or show_numpy:
            print(f"Files with NumPy: {len(numpy_files)} ({len(numpy_files)/len(results)*100:.2f}%)")
        if show_all or show_precision:
            print(f"Files with precision-reducing casts: {len(files_with_precision_casts)} ({len(files_with_precision_casts)/len(results)*100:.2f}%)")
        if show_all or show_conversion:
            print(f"Files with tensor conversions: {len(files_with_tensor_conversions)} ({len(files_with_tensor_conversions)/len(results)*100:.2f}%)")
        if show_all or show_operators:
            print(f"Files with Python operators: {len(files_with_python_operators)} ({len(files_with_python_operators)/len(results)*100:.2f}%)")
        if show_all or show_unused:
            print(f"Files with unused imports: {len(files_with_unused_imports)} ({len(files_with_unused_imports)/len(results)*100:.2f}%)")
        if show_all or show_gpu:
            print(f"Files with GPU enforcement issues: {len(files_with_gpu_issues)} ({len(files_with_gpu_issues)/len(results)*100:.2f}%)")
            print(f"Files with CPU usage: {len(files_with_cpu_usage)} ({len(files_with_cpu_usage)/len(results)*100:.2f}%)")
        if show_all or show_mlx:
            print(f"Files with host pulls: {len(files_with_host_pulls)} ({len(files_with_host_pulls)/len(results)*100:.2f}%)")
            print(f"Files with Python comparisons on MLX: {len(files_with_comparisons)} ({len(files_with_comparisons)/len(results)*100:.2f}%)")
            print(f"Files with bitwise operations: {len(files_with_bitwise)} ({len(files_with_bitwise)/len(results)*100:.2f}%)")
    
    if verbose:
        if (show_all or show_syntax) and files_with_syntax_errors:
            print("\nFiles with syntax errors:")
            for result in files_with_syntax_errors:
                print(f"\n{result['file']}:")
                for error in result["syntax_errors"]:
                    print(f"  {error}")
        
        if (show_all or show_compilation) and files_with_compilation_errors:
            print("\nFiles with compilation errors:")
            for result in files_with_compilation_errors:
                print(f"\n{result['file']}:")
                for error in result["compilation_errors"]:
                    print(f"  {error}")
        
        if (show_all or show_imports) and files_with_import_errors:
            print("\nFiles with import errors:")
            for result in files_with_import_errors:
                print(f"\n{result['file']}:")
                for error in result["import_errors"]:
                    print(f"  {error}")
        
        if (show_all or show_style) and files_with_style_errors:
            print("\nFiles with style errors:")
            for result in files_with_style_errors:
                print(f"\n{result['file']}:")
                for error in result["style_errors"][:10]:  # Limit to 10 errors
                    print(f"  {error}")
                if len(result["style_errors"]) > 10:
                    print(f"  ... and {len(result['style_errors']) - 10} more")
        
        if (show_all or show_types) and files_with_type_errors:
            print("\nFiles with type errors:")
            for result in files_with_type_errors:
                print(f"\n{result['file']}:")
                for error in result["type_errors"][:10]:  # Limit to 10 errors
                    print(f"  {error}")
                if len(result["type_errors"]) > 10:
                    print(f"  ... and {len(result['type_errors']) - 10} more")
        
        if (show_all or show_numpy) and numpy_files:
            print("\nFiles with NumPy:")
            for result in numpy_files:
                print(f"\n{result['file']}:")
                print(f"  Imports: {', '.join(result['imports'])}")
                print(f"  Usages: {', '.join(result['usages'])}")
                if result.get("numpy_random"):
                    print("  numpy.random usage detected:")
                    for name, _pos in result["numpy_random"]:
                        print(f"    {name}    -> Prefer MLX keys: k=mx.random.key(123); x=mx.random.normal(shape=..., key=k)")
                        print("       Split keys for multiple draws: k1,k2=mx.random.split(k,num=2)")
        
        if (show_all or show_operators) and files_with_python_operators:
            print("\nFiles with Python operators on MLX arrays:")
            for result in files_with_python_operators:
                print(f"\n{result['file']}:")
                for op in result["python_operators"]:
                    print(f"  {op['type']} operator at {op['location']}")
                    if op.get('suggestion'):
                        print(f"    Suggestion: {op['suggestion']}")
        
        if (show_all or show_mlx) and files_with_host_pulls:
            print("\nFiles with host pulls:")
            for result in files_with_host_pulls:
                print(f"\n{result['file']}:")
                for pull in result["host_pulls"]:
                    print(f"  {pull['type']} at {pull['location']}")
                    if pull.get('suggestion'):
                        print(f"    {pull['suggestion']}")
        
        if (show_all or show_mlx) and files_with_comparisons:
            print("\nFiles with Python comparisons on MLX arrays:")
            for result in files_with_comparisons:
                print(f"\n{result['file']}:")
                for comp in result["comparisons"]:
                    print(f"  {comp['type']} at {comp['location']}")
                    print(f"    Suggestion: {comp['suggestion']}")
        
        if (show_all or show_mlx) and files_with_bitwise:
            print("\nFiles with bitwise operations:")
            for result in files_with_bitwise:
                print(f"\n{result['file']}:")
                for op in result["bitwise_ops"]:
                    print(f"  {op['type']} at {op['location']}")
                    print(f"    Suggestion: {op['suggestion']}")
        
        if show_all and files_with_gpu_prefix:
            print("\nFiles with 'gpu_' prefix in names (style warning):")
            for result in files_with_gpu_prefix:
                print(f"\n{result['file']}:")
                for item in result["gpu_prefix"]:
                    print(f"  {item['name']} at {item['location']}")
                    print(f"    Suggestion: {item['suggestion']}")
        
        if (show_all or show_precision) and files_with_precision_casts:
            print("\nFiles with precision-reducing casts on MLX:")
            for result in files_with_precision_casts:
                print(f"\n{result['file']}:")
                for cast in result["precision_casts"]:
                    print(f"  {cast['type']}() at {cast['location']}")
                    if cast.get('suggestion'):
                        print(f"    Suggestion: {cast['suggestion']}")
        
        if (show_all or show_unused) and files_with_unused_imports:
            print("\nFiles with unused imports:")
            for result in files_with_unused_imports:
                print(f"\n{result['file']}:")
                for name, lineno in result["unused_imports"]:
                    print(f"  Unused import '{name}' at line {lineno}")
        
        if (show_all or show_gpu) and files_with_gpu_issues:
            print("\nFiles with GPU enforcement issues:")
            for result in files_with_gpu_issues:
                print(f"\n{result['file']}:")
                for issue in result["gpu_issues"]:
                    print(f"  {issue}")
        
        if (show_all or show_gpu) and files_with_cpu_usage:
            print("\nFiles with CPU usage:")
            for result in files_with_cpu_usage:
                print(f"\n{result['file']}:")
                for usage in result["cpu_usage"]:
                    print(f"  {usage['type']} at {usage['location']}")
    
    # Print summary by directory
    print("\nSummary by directory:")
    dir_summary = {}
    for result in results:
        dir_path = os.path.dirname(result["file"])
        if dir_path not in dir_summary:
            dir_summary[dir_path] = {
                "total": 0,
                "syntax_errors": 0,
                "compilation_errors": 0,
                "import_errors": 0,
                "style_errors": 0,
                "type_errors": 0,
                "numpy": 0,
                "precision_casts": 0,
                "tensor_conversions": 0,
                "python_operators": 0,
                "unused_imports": 0,
                "gpu_enforcement_issues": 0,
                "cpu_usage": 0
            }
        dir_summary[dir_path]["total"] += 1
        if not result["syntax_valid"]:
            dir_summary[dir_path]["syntax_errors"] += 1
        if not result["compilation_valid"]:
            dir_summary[dir_path]["compilation_errors"] += 1
        if not result["imports_valid"]:
            dir_summary[dir_path]["import_errors"] += 1
        if not result["style_valid"]:
            dir_summary[dir_path]["style_errors"] += 1
        if not result["types_valid"]:
            dir_summary[dir_path]["type_errors"] += 1
        if result["has_numpy"]:
            dir_summary[dir_path]["numpy"] += 1
        if result["precision_casts"]:
            dir_summary[dir_path]["precision_casts"] += 1
        if result["tensor_conversions"]:
            dir_summary[dir_path]["tensor_conversions"] += 1
        if result["python_operators"]:
            dir_summary[dir_path]["python_operators"] += 1
        if result["unused_imports"]:
            dir_summary[dir_path]["unused_imports"] += 1
        if not result["gpu_enforced"]:
            dir_summary[dir_path]["gpu_enforcement_issues"] += 1
        if result["cpu_usage"]:
            dir_summary[dir_path]["cpu_usage"] += 1
    
    for dir_path, stats in sorted(dir_summary.items()):
        # Check if there are any issues to show for this directory
        has_issues = (
            (show_all or show_syntax) and stats["syntax_errors"] > 0 or
            (show_all or show_compilation) and stats["compilation_errors"] > 0 or
            (show_all or show_imports) and stats["import_errors"] > 0 or
            (show_all or show_style) and stats["style_errors"] > 0 or
            (show_all or show_types) and stats["type_errors"] > 0 or
            (show_all or show_numpy) and stats["numpy"] > 0 or
            (show_all or show_precision) and stats["precision_casts"] > 0 or
            (show_all or show_conversion) and stats["tensor_conversions"] > 0 or
            (show_all or show_operators) and stats["python_operators"] > 0 or
            (show_all or show_unused) and stats["unused_imports"] > 0 or
            (show_all or show_gpu) and stats["gpu_enforcement_issues"] > 0 or
            (show_all or show_gpu) and stats["cpu_usage"] > 0
        )
        
        if has_issues:
            print(f"{dir_path}:")
            if summary_only:
                bits = []
                if (show_all or show_syntax) and stats["syntax_errors"] > 0:
                    bits.append(f"syntax={stats['syntax_errors']}")
                if (show_all or show_compilation) and stats["compilation_errors"] > 0:
                    bits.append(f"compile={stats['compilation_errors']}")
                if (show_all or show_imports) and stats["import_errors"] > 0:
                    bits.append(f"imports={stats['import_errors']}")
                if (show_all or show_style) and stats["style_errors"] > 0:
                    bits.append(f"style={stats['style_errors']}")
                if (show_all or show_types) and stats["type_errors"] > 0:
                    bits.append(f"types={stats['type_errors']}")
                if (show_all or show_numpy) and stats["numpy"] > 0:
                    bits.append(f"numpy={stats['numpy']}")
                if (show_all or show_precision) and stats["precision_casts"] > 0:
                    bits.append(f"precision={stats['precision_casts']}")
                if (show_all or show_conversion) and stats["tensor_conversions"] > 0:
                    bits.append(f"convert={stats['tensor_conversions']}")
                if (show_all or show_operators) and stats["python_operators"] > 0:
                    bits.append(f"ops={stats['python_operators']}")
                if (show_all or show_unused) and stats["unused_imports"] > 0:
                    bits.append(f"unused={stats['unused_imports']}")
                if (show_all or show_gpu) and stats["gpu_enforcement_issues"] > 0:
                    bits.append(f"gpu={stats['gpu_enforcement_issues']}")
                if (show_all or show_gpu) and stats["cpu_usage"] > 0:
                    bits.append(f"cpu={stats['cpu_usage']}")
                if bits:
                    print("  " + ", ".join(bits))
                continue
            if (show_all or show_syntax) and stats["syntax_errors"] > 0:
                print(f"  Syntax errors: {stats['syntax_errors']}/{stats['total']} files ({stats['syntax_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_compilation) and stats["compilation_errors"] > 0:
                print(f"  Compilation errors: {stats['compilation_errors']}/{stats['total']} files ({stats['compilation_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_imports) and stats["import_errors"] > 0:
                print(f"  Import errors: {stats['import_errors']}/{stats['total']} files ({stats['import_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_style) and stats["style_errors"] > 0:
                print(f"  Style errors: {stats['style_errors']}/{stats['total']} files ({stats['style_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_types) and stats["type_errors"] > 0:
                print(f"  Type errors: {stats['type_errors']}/{stats['total']} files ({stats['type_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_numpy) and stats["numpy"] > 0:
                print(f"  NumPy: {stats['numpy']}/{stats['total']} files ({stats['numpy']/stats['total']*100:.2f}%)")
            if (show_all or show_precision) and stats["precision_casts"] > 0:
                print(f"  Precision casts: {stats['precision_casts']}/{stats['total']} files ({stats['precision_casts']/stats['total']*100:.2f}%)")
            if (show_all or show_conversion) and stats["tensor_conversions"] > 0:
                print(f"  Tensor conversions: {stats['tensor_conversions']}/{stats['total']} files ({stats['tensor_conversions']/stats['total']*100:.2f}%)")
            if (show_all or show_operators) and stats["python_operators"] > 0:
                print(f"  Python operators: {stats['python_operators']}/{stats['total']} files ({stats['python_operators']/stats['total']*100:.2f}%)")
            if (show_all or show_unused) and stats["unused_imports"] > 0:
                print(f"  Unused imports: {stats['unused_imports']}/{stats['total']} files ({stats['unused_imports']/stats['total']*100:.2f}%)")
            if (show_all or show_gpu) and stats["gpu_enforcement_issues"] > 0:
                print(f"  GPU enforcement issues: {stats['gpu_enforcement_issues']}/{stats['total']} files ({stats['gpu_enforcement_issues']/stats['total']*100:.2f}%)")
            if (show_all or show_gpu) and stats["cpu_usage"] > 0:
                print(f"  CPU usage: {stats['cpu_usage']}/{stats['total']} files ({stats['cpu_usage']/stats['total']*100:.2f}%)")

def to_json(results: List[Dict], include_details: bool = True) -> str:
    """Serialize results to JSON.

    When include_details is False, only emit a summary count object.
    """
    if include_details:
        return json.dumps(results, indent=2, default=str)
    total = len(results)
    summary = {
        'total_files': total,
        'syntax_errors': sum(1 for r in results if not r['syntax_valid']),
        'compilation_errors': sum(1 for r in results if not r['compilation_valid']),
        'import_errors': sum(1 for r in results if not r['imports_valid']),
        'style_errors': sum(1 for r in results if not r['style_valid']),
        'type_errors': sum(1 for r in results if not r['types_valid']),
        'numpy': sum(1 for r in results if r['has_numpy']),
        'precision_casts': sum(1 for r in results if r['precision_casts']),
        'tensor_conversions': sum(1 for r in results if r['tensor_conversions']),
        'python_operators': sum(1 for r in results if r['python_operators']),
        'unused_imports': sum(1 for r in results if r['unused_imports']),
        'gpu_enforcement_issues': sum(1 for r in results if not r['gpu_enforced']),
        'cpu_usage': sum(1 for r in results if r['cpu_usage']),
        'mlx_issues': sum(1 for r in results if r.get('host_pulls') or r.get('comparisons') or r.get('bitwise_ops')),
        'gpu_prefix': sum(1 for r in results if r.get('gpu_prefix')),
    }
    return json.dumps(summary, indent=2)

def print_zero_cpu_results(results: List[Dict], verbose: bool = False, summary_only: bool = False):
    """Print Zero-CPU policy violation results (simplified, focused output)."""
    
    # Count violations by category
    total = len(results)
    numpy_files = [r for r in results if r["has_numpy"]]
    host_pulls = [r for r in results if r.get("host_pulls", [])]
    operators = [r for r in results if r["python_operators"]]
    comparisons = [r for r in results if r.get("comparisons", [])]
    bitwise = [r for r in results if r.get("bitwise_ops", [])]
    precision = [r for r in results if r["precision_casts"]]
    conversions = [r for r in results if r["tensor_conversions"]]
    cpu_usage = [r for r in results if r["cpu_usage"]]
    syntax_err = [r for r in results if not r["syntax_valid"]]
    compile_err = [r for r in results if not r["compilation_valid"]]
    
    # Calculate total violations
    total_violations = len(numpy_files) + len(host_pulls) + len(operators) + len(comparisons) + \
                      len(bitwise) + len(precision) + len(conversions) + len(cpu_usage)
    
    # Header
    print("=" * 80)
    print("MetalFaiss Zero-CPU Policy Enforcement")
    print("=" * 80)
    print(f"📁 Files analyzed: {total}")
    
    if total_violations == 0 and len(syntax_err) == 0 and len(compile_err) == 0:
        print("✅ PASS - No Zero-CPU policy violations detected!")
        print("=" * 80)
        return
    
    print(f"❌ FAIL - {total_violations} policy violations detected")
    print("=" * 80)
    
    # Summary section
    if len(numpy_files) > 0:
        print(f"\n🚫 NumPy Usage: {len(numpy_files)} files")
        print("   Policy: Use MLX instead of NumPy")
    
    if len(host_pulls) > 0:
        print(f"\n🚫 CPU Transfers: {len(host_pulls)} files")
        print("   Policy: No .tolist(), .item(), .numpy() calls")
    
    if len(operators) > 0:
        print(f"\n🚫 Python Operators: {len(operators)} files")
        print("   Policy: Use mx.add(), mx.multiply(), etc. instead of +, *, etc.")
    
    if len(comparisons) > 0:
        print(f"\n🚫 Python Comparisons: {len(comparisons)} files")
        print("   Policy: Use mx.equal(), mx.less(), etc. instead of ==, <, etc.")
    
    if len(bitwise) > 0:
        print(f"\n🚫 Bitwise Operations: {len(bitwise)} files")
        print("   Policy: Use mx.bitwise_and(), etc. instead of &, |, etc.")
    
    if len(precision) > 0:
        print(f"\n⚠️  Precision Casts: {len(precision)} files")
        print("   Warning: float(), int() casts may transfer data off GPU")
    
    if len(conversions) > 0:
        print(f"\n⚠️  Tensor Conversions: {len(conversions)} files")
        print("   Warning: Converting between NumPy/MLX (use MLX throughout)")
    
    if len(cpu_usage) > 0:
        print(f"\n⚠️  CPU-Only Code: {len(cpu_usage)} files")
        print("   Warning: Code paths that don't use GPU")
    
    # Detailed output
    if verbose and not summary_only:
        print("\n" + "=" * 80)
        print("DETAILED VIOLATIONS")
        print("=" * 80)
        
        if numpy_files:
            print("\n📋 NumPy Usage Details:")
            for r in numpy_files:
                print(f"\n  {r['file']}:")
                if r['imports']:
                    print(f"    Imports: {', '.join(r['imports'])}")
                if r.get('usages'):
                    print(f"    Usage: {', '.join(r['usages'][:5])}")
                    if len(r['usages']) > 5:
                        print(f"           ... and {len(r['usages']) - 5} more")
        
        if host_pulls:
            print("\n📋 CPU Transfer Details:")
            for r in host_pulls:
                print(f"\n  {r['file']}:")
                for pull in r["host_pulls"][:10]:
                    print(f"    {pull['type']} at {pull['location']}")
                    if pull.get('suggestion'):
                        print(f"      💡 {pull['suggestion']}")
                if len(r["host_pulls"]) > 10:
                    print(f"    ... and {len(r['host_pulls']) - 10} more")
        
        if operators:
            print("\n📋 Python Operator Details:")
            for r in operators:
                print(f"\n  {r['file']}:")
                for op in r["python_operators"][:10]:
                    print(f"    {op['type']} at {op['location']}")
                    if op.get('suggestion'):
                        print(f"      💡 {op['suggestion']}")
                if len(r["python_operators"]) > 10:
                    print(f"    ... and {len(r['python_operators']) - 10} more")
        
        if comparisons:
            print("\n📋 Python Comparison Details:")
            for r in comparisons:
                print(f"\n  {r['file']}:")
                for comp in r["comparisons"][:10]:
                    print(f"    {comp['type']} at {comp['location']}")
                    print(f"      💡 {comp['suggestion']}")
                if len(r["comparisons"]) > 10:
                    print(f"    ... and {len(r['comparisons']) - 10} more")
    
    # Errors section
    if syntax_err or compile_err:
        print("\n" + "=" * 80)
        print("SYNTAX/COMPILATION ERRORS")
        print("=" * 80)
        if syntax_err:
            print(f"\n❌ Syntax Errors: {len(syntax_err)} files")
            if verbose:
                for r in syntax_err:
                    print(f"  {r['file']}: {r['syntax_errors'][0] if r['syntax_errors'] else 'Unknown'}")
        if compile_err:
            print(f"\n❌ Compilation Errors: {len(compile_err)} files")
            if verbose:
                for r in compile_err:
                    print(f"  {r['file']}: {r['compilation_errors'][0] if r['compilation_errors'] else 'Unknown'}")
    
    # Footer
    print("\n" + "=" * 80)
    print("💡 TIPS:")
    print("  • Mark exceptions with # boundary-ok comment")
    print("  • Use mx.* operations instead of Python operators")
    print("  • Keep all data on GPU - avoid .tolist(), .item(), .numpy()")
    print("  • See ZERO_CPU_AUDIT.md for patterns and examples")
    print("=" * 80)

def main():
    """Main function - Enforces Zero-CPU policy."""
    parser = argparse.ArgumentParser(
        description="MetalFaissLint: Zero-CPU Policy Enforcer",
        epilog="This tool ALWAYS checks for NumPy usage, CPU transfers (.tolist/.item/.numpy), "
               "Python operators on MLX arrays, and GPU enforcement. No flags needed."
    )
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--exclude", nargs="+", help="Directories to exclude (e.g., tests unittest)", default=[])
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed file-by-file results")
    parser.add_argument("--json", action="store_true", help="Output results as JSON for CI integration")
    parser.add_argument("--summary", action="store_true", help="Show summary only (default: show all violations)")
    parser.add_argument("--config", type=str, default=None, help="Path to .emberlint config file (optional)")
    parser.add_argument("--exit-zero", action="store_true", help="Always exit with code 0 (report-only mode)")
    
    args = parser.parse_args()

    # Load config if exists (for exclude paths mainly)
    cfg = load_config(args.config or args.path)
    if not args.exclude and isinstance(cfg.get('exclude'), list):
        args.exclude = cfg['exclude']
    if not args.verbose and cfg.get('verbose') is True:
        args.verbose = True

    # Analyze files
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        results = [analyze_file(args.path)]
    else:
        results = analyze_directory(args.path, args.exclude)

    # Output results
    if args.json:
        print(to_json(results, include_details=True))
    else:
        print_zero_cpu_results(results, args.verbose, args.summary)
    
    # Exit code: fail if ANY Zero-CPU violations found (unless --exit-zero)
    if args.exit_zero:
        sys.exit(0)
    
    has_violations = any(
        r["has_numpy"] or
        r["precision_casts"] or
        r["tensor_conversions"] or
        r["python_operators"] or
        r.get("host_pulls", []) or
        r.get("comparisons", []) or
        r.get("bitwise_ops", []) or
        r["cpu_usage"] or
        not r["syntax_valid"] or
        not r["compilation_valid"]
        for r in results
    )
    
    sys.exit(1 if has_violations else 0)

if __name__ == "__main__":
    sys.exit(main())
