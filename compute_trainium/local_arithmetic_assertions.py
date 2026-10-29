"""Conservative local arithmetic assertions, preserving original source labels.

This verifies a bounded syntactic computation, not execution of an entire file.
Only local constant assignments and pure integer expressions are interpreted.
Unknown statements forget all bindings. Source code is never executed.
"""
import ast
import re

from .authored_code_assertions import Unsupported, Value, expression, literal
from .natural_code_context import node_span


def verified_local_assertions(source):
    if len(source) > 100000:
        raise Unsupported('source_length')
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 12000:
        raise Unsupported('AST_size')
    rows = []

    def scope(statements):
        values, origins = {}, {}

        def evaluate(node):
            value = expression(node, values)
            used = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
            start = min([node_span(source, node)[0]]+[origins[n] for n in used])
            return value, start

        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope(statement.body)
                values.clear()
                origins.clear()
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and type(statement.value.value) is str:
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                try:
                    value, start = evaluate(statement.value)
                except Unsupported:
                    values.clear()
                    origins.clear()
                    continue
                name = statement.targets[0].id
                values[name], origins[name] = value, min(start, node_span(source, statement)[0])
                continue
            if not isinstance(statement, ast.Assert):
                values.clear()
                origins.clear()
                continue
            test = statement.test
            if (statement.msg is not None or not isinstance(test, ast.Compare) or len(test.ops) != 1
                    or not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1):
                values.clear()
                origins.clear()
                continue
            try:
                value, start = evaluate(test.left)
                expected = literal(test.comparators[0])
            except Unsupported:
                values.clear()
                origins.clear()
                continue
            if value.integer != expected:
                # A failed assertion blocks this straight-line continuation.
                break
            target_start, target_end = node_span(source, test.comparators[0])
            if (value.operations < 2 or expected in value.literals
                    or re.search(r'(?<![\w.])'+re.escape(str(expected))+r'(?![\w.])', source[start:target_start])):
                continue
            rows.append(dict(target_start=target_start, target_end=target_end,
                             premise_start=start, premise_end=target_start,
                             operations=value.operations,
                             dependency_names=sum(isinstance(n, ast.Name) for n in ast.walk(test.left))))

    scope(tree.body)
    return rows
