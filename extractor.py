import ast
import math

FEATURE_COLS = [
    'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
    'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
    'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount',
]

_SAFE_LOG2_MIN = 2


def _safe_log2(x: float) -> float:
    return math.log2(max(float(x), _SAFE_LOG2_MIN))


def _clamp_metrics(metrics: dict) -> dict:
    bounds = {
        'loc': (0, 50000), 'lOCode': (0, 50000), 'lOComment': (0, 10000),
        'lOBlank': (0, 10000), 'locCodeAndComment': (0, 5000),
        'v(g)': (1, 500), 'ev(g)': (1, 500), 'iv(g)': (1, 500),
        'branchCount': (0, 2000),
        'n': (0, 100000), 'v': (0, 500000), 'l': (0.0, 1.0),
        'd': (0, 5000), 'i': (0, 100000), 'e': (0, 10_000_000),
        'b': (0, 500), 't': (0, 1_000_000),
        'uniq_Op': (0, 500), 'uniq_Opnd': (0, 10000),
        'total_Op': (0, 100000), 'total_Opnd': (0, 100000),
    }
    for col, (lo, hi) in bounds.items():
        if col in metrics:
            metrics[col] = float(max(lo, min(hi, metrics[col])))
    return metrics


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_complexities = []
        self.branch_count = 0
        self._stack = []

    def _enter_function(self, node):
        self._stack.append(1)
        self.generic_visit(node)
        self.function_complexities.append(self._stack.pop())

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    def _add_branch(self, node, weight=1):
        self.branch_count += weight
        if self._stack:
            self._stack[-1] += weight
        self.generic_visit(node)

    def visit_If(self, node):
        self._add_branch(node)

    def visit_While(self, node):
        self._add_branch(node)

    def visit_For(self, node):
        self._add_branch(node)

    def visit_ExceptHandler(self, node):
        self._add_branch(node)

    def visit_With(self, node):
        self._add_branch(node)

    def visit_Assert(self, node):
        self._add_branch(node)

    def visit_IfExp(self, node):
        self._add_branch(node)

    def visit_BoolOp(self, node):
        weight = max(len(node.values) - 1, 0)
        self._add_branch(node, weight)

    def visit_Match(self, node):
        weight = max(len(node.cases) - 1, 0)
        self._add_branch(node, weight)


class _HalsteadVisitor(ast.NodeVisitor):
    def __init__(self):
        self.operators = []
        self.operands = []

    def visit_BinOp(self, node):
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self.operators.append(type(op).__name__)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.operators.append(type(node.op).__name__ + '=')
        self.generic_visit(node)

    def visit_Call(self, node):
        self.operators.append('call')
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.operators.append('=')
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.operators.append('.')
        self.operands.append(node.attr)
        self.visit(node.value)

    def visit_Subscript(self, node):
        self.operators.append('[]')
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id not in ('True', 'False', 'None'):
            self.operands.append(node.id)

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float, complex)):
            self.operands.append('__NUM__')
        elif isinstance(v, str) and v.strip():
            self.operands.append('__STR__')

    def visit_FunctionDef(self, node):
        self.operands.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.operands.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.operands.append(node.name)
        self.generic_visit(node)


def extract_metrics(source_code: str) -> dict:
    source_code = source_code.lstrip('\ufeff')
    metrics = {col: 0.0 for col in FEATURE_COLS}

    try:
        lines = source_code.splitlines()
        loc = len(lines)
        blank_lines = sum(1 for ln in lines if not ln.strip())
        comment_lines = sum(1 for ln in lines if ln.strip().startswith('#'))
        code_and_comment = sum(
            1 for ln in lines
            if ln.strip() and '#' in ln and not ln.strip().startswith('#')
        )
        code_lines = max(loc - blank_lines - comment_lines, 0)
        metrics['loc'] = float(loc)
        metrics['lOCode'] = float(code_lines)
        metrics['lOComment'] = float(comment_lines)
        metrics['lOBlank'] = float(blank_lines)
        metrics['locCodeAndComment'] = float(code_and_comment)
    except Exception:
        pass

    try:
        tree = ast.parse(source_code)

        cv = _ComplexityVisitor()
        cv.visit(tree)
        fc = cv.function_complexities

        if fc:
            total_vg = float(sum(fc))
            metrics['v(g)'] = total_vg
            metrics['ev(g)'] = float(max(fc))
            metrics['iv(g)'] = float(total_vg / len(fc))
        else:
            metrics['v(g)'] = 1.0
            metrics['ev(g)'] = 1.0
            metrics['iv(g)'] = 1.0

        metrics['branchCount'] = float(cv.branch_count)

    except SyntaxError:
        metrics['v(g)'] = 0.0
        metrics['ev(g)'] = 0.0
        metrics['iv(g)'] = 0.0
        metrics['branchCount'] = 0.0
    except Exception:
        pass

    try:
        tree = ast.parse(source_code)
        hv = _HalsteadVisitor()
        hv.visit(tree)

        ops, opnds = hv.operators, hv.operands
        n1 = max(len(set(ops)), 1)
        n2 = max(len(set(opnds)), 1)
        N1 = max(len(ops), 1)
        N2 = max(len(opnds), 1)

        n = n1 + n2
        N = N1 + N2
        V = N * _safe_log2(n)
        D = (n1 / 2.0) * (N2 / n2)
        L = 1.0 / max(D, 0.001)
        L = min(L, 1.0)
        I = L * V
        E = D * V
        B = V / 3000.0
        T = E / 18.0

        metrics['n'] = float(N)
        metrics['v'] = float(V)
        metrics['l'] = float(L)
        metrics['d'] = float(D)
        metrics['i'] = float(I)
        metrics['e'] = float(E)
        metrics['b'] = float(B)
        metrics['t'] = float(T)
        metrics['uniq_Op'] = float(n1)
        metrics['uniq_Opnd'] = float(n2)
        metrics['total_Op'] = float(N1)
        metrics['total_Opnd'] = float(N2)

    except SyntaxError:
        pass
    except Exception:
        pass

    return _clamp_metrics(metrics)


def get_code_summary(source_code: str) -> dict:
    source_code = source_code.lstrip('\ufeff')
    summary = {
        'functions': [],
        'classes': [],
        'imports': [],
        'total_lines': 0,
        'docstrings': 0,
        'async_functions': 0,
        'decorators': 0,
        'comprehensions': 0,
        'error': None,
    }
    try:
        summary['total_lines'] = len(source_code.splitlines())
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                summary['functions'].append(node.name)
                if isinstance(node, ast.AsyncFunctionDef):
                    summary['async_functions'] += 1
                summary['decorators'] += len(node.decorator_list)
                if ast.get_docstring(node):
                    summary['docstrings'] += 1
            elif isinstance(node, ast.ClassDef):
                summary['classes'].append(node.name)
                if ast.get_docstring(node):
                    summary['docstrings'] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    summary['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    summary['imports'].append(node.module)
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                summary['comprehensions'] += 1
    except SyntaxError as e:
        summary['error'] = f"SyntaxError: {e}"
    except Exception as e:
        summary['error'] = str(e)
    return summary


def get_risk_breakdown(metrics: dict) -> list:
    risks = []

    vg = metrics.get('v(g)', 0)
    if vg > 20:
        risks.append({'factor': 'Cyclomatic Complexity', 'value': vg, 'severity': 'high',
                      'message': f'Complexity {vg:.0f} far exceeds safe threshold of 10'})
    elif vg > 10:
        risks.append({'factor': 'Cyclomatic Complexity', 'value': vg, 'severity': 'medium',
                      'message': f'Complexity {vg:.0f} exceeds recommended threshold of 10'})

    bc = metrics.get('branchCount', 0)
    if bc > 50:
        risks.append({'factor': 'Branch Count', 'value': bc, 'severity': 'high',
                      'message': f'{bc:.0f} branches — highly branched control flow is hard to test'})
    elif bc > 20:
        risks.append({'factor': 'Branch Count', 'value': bc, 'severity': 'medium',
                      'message': f'{bc:.0f} branches — consider simplifying conditionals'})

    v = metrics.get('v', 0)
    if v > 1000:
        risks.append({'factor': 'Halstead Volume', 'value': v, 'severity': 'high',
                      'message': f'Volume {v:.0f} indicates extremely dense logic'})
    elif v > 500:
        risks.append({'factor': 'Halstead Volume', 'value': v, 'severity': 'medium',
                      'message': f'Volume {v:.0f} indicates complex logic'})

    b = metrics.get('b', 0)
    if b > 0.5:
        risks.append({'factor': 'Estimated Bugs', 'value': b, 'severity': 'high',
                      'message': f'Halstead estimates {b:.2f} bugs in this module'})
    elif b > 0.25:
        risks.append({'factor': 'Estimated Bugs', 'value': b, 'severity': 'medium',
                      'message': f'Halstead estimates {b:.2f} potential bugs'})

    loc = metrics.get('loc', 0)
    if loc > 500:
        risks.append({'factor': 'Lines of Code', 'value': loc, 'severity': 'medium',
                      'message': f'{loc:.0f} lines — consider splitting into smaller modules'})

    d = metrics.get('d', 0)
    if d > 30:
        risks.append({'factor': 'Halstead Difficulty', 'value': d, 'severity': 'high',
                      'message': f'Difficulty {d:.1f} — very hard to understand and maintain'})
    elif d > 15:
        risks.append({'factor': 'Halstead Difficulty', 'value': d, 'severity': 'medium',
                      'message': f'Difficulty {d:.1f} — moderately complex logic'})

    return risks