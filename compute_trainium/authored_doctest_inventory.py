"""Syntax-only census of existing doctest inputs and literal authored outputs.

Nothing here executes source code, verifies an answer, or constructs a task.
Missing dependencies and outputs are counted, never supplied or inferred.
"""
import ast
from collections import Counter
import doctest
import re


BUILTINS = frozenset({'abs', 'all', 'any', 'bool', 'int', 'len', 'list', 'max',
                     'min', 'range', 'reversed', 'round', 'sorted', 'sum', 'tuple'})
LEAVES = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow,
          ast.Div, ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
          ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
          ast.In, ast.NotIn, ast.Load, ast.Store)
NODES = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name,
         ast.Constant, ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp,
         ast.BoolOp, ast.Compare, ast.Subscript, ast.Slice, ast.Call,
         ast.keyword, *LEAVES)


def literal_kind(text):
    """Inspect a literal AST, without evaluating an expression or constructor."""
    if len(text) > 80 or len(text.strip().splitlines()) != 1:
        return None
    try:
        node = ast.parse(text.strip(), mode='eval').body
    except (SyntaxError, ValueError, RecursionError):
        return None
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return 'boolean'
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10**9:
        return 'integer'
    return None


def closed_operations(tree, known):
    """Conservative syntax coverage; this is NOT semantic verification.

    Track expression provenance through simple assignments, rejecting imports,
    attributes, calls except named primitives, unknown reads and shadowed calls.
    A failed example invalidates the entire current environment.
    """
    nodes = list(ast.walk(tree))
    if len(nodes) > 128 or any(not isinstance(n, NODES) for n in nodes):
        return None
    env = dict(known)

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError('unknown read')
            return env[node.id]
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, bool) or abs(node.value) > 10**9:
                raise ValueError('unsupported literal')
            return 0
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in BUILTINS or node.func.id in env:
                raise ValueError('unknown or shadowed call')
            return 1 + sum(expression(a) for a in [*node.args, *(k.value for k in node.keywords)])
        if isinstance(node, (ast.List, ast.Tuple)):
            return sum(expression(a) for a in node.elts)
        if isinstance(node, ast.Slice):
            return sum(expression(a) for a in (node.lower, node.upper, node.step) if a is not None)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Subscript)):
            return 1 + sum(expression(a) for a in ast.iter_child_nodes(node)
                           if not isinstance(a, LEAVES))
        raise ValueError('unsupported expression')

    last, reads_prior = None, False
    try:
        for statement in tree.body:
            reads_prior |= any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                               and n.id in known for n in ast.walk(statement))
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise ValueError('complex assignment')
                env[statement.targets[0].id] = expression(statement.value)
                last = None
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                key = statement.target.id
                if key not in env:
                    raise ValueError('unknown augmented target')
                env[key] += 1 + expression(statement.value)
                last = None
            elif isinstance(statement, ast.Expr):
                last = expression(statement.value)
            else:
                raise ValueError('unsupported statement')
        if any(v > 64 for v in env.values()) or (last is not None and last > 64):
            return None
    except ValueError:
        return None
    return env, last, reads_prior


def inventory(source):
    if len(source) > 100000:
        raise ValueError('source_length')
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 12000:
        raise ValueError('AST_size')
    counts = Counter()
    # Every original string literal is isolated: no scope carries across docs.
    for node in nodes:
        if not isinstance(node, ast.Constant) or type(node.value) is not str or '>>>' not in node.value:
            continue
        raw = ast.get_source_segment(source, node)
        if not raw:
            continue
        opening = re.match(r"(?i)^[rub]*(\"\"\"|''')", raw)
        if opening is None or not raw.endswith(opening.group(1)):
            counts['non_multiline_literal_exclusion'] += 1
            continue
        # Keep original contents; do not decode escapes or let the closing
        # Python delimiter become part of the authored doctest output.
        raw = raw[opening.end():-3]
        try:
            examples = doctest.DocTestParser().get_examples(raw)
        except ValueError:
            counts['malformed_doctest_literal'] += 1
            continue
        counts['doctest_literals'] += 1
        env, setup_start, previous_end = {}, None, None
        lines = raw.splitlines(keepends=True)
        for example in examples:
            # Prose/blank separation may describe a new environment. Reset
            # conservatively rather than inventing continuity across examples.
            if previous_end is not None and example.lineno > previous_end:
                env, setup_start = {}, None
            previous_end = example.lineno + len(example.source.splitlines()) + len(example.want.splitlines())
            counts['prompt_examples'] += 1
            kind = literal_kind(example.want)
            if kind:
                counts[kind + '_authored_outputs'] += 1
            try:
                parsed = ast.parse(example.source)
                checked = closed_operations(parsed, env)
            except (SyntaxError, ValueError, RecursionError):
                checked = None
            if checked is None or example.exc_msg or example.options:
                counts['unsupported_or_context_open'] += 1
                env, setup_start = {}, None
                continue
            if setup_start is None:
                setup_start = example.lineno
            env, operations, reads_prior = checked
            counts['closed_grammar_examples'] += 1
            if kind and operations is not None:
                counts['unverified_' + kind + '_closed_outputs'] += 1
                if operations >= 2:
                    counts['unverified_multistep_outputs'] += 1
                    counts['unverified_multistep_prior_variable'] += int(reads_prior)
                    chars = len(''.join(lines[setup_start:previous_end]))
                    counts['unverified_multistep_under_2048_characters'] += int(chars <= 2048)
    return counts
