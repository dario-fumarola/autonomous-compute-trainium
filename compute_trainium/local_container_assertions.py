"""Verify authored list-length assertions after local literal mutations.

This is a deliberately small static language, never Python execution. Aliases,
branches, external calls and unknown statements invalidate the environment.
Only original assertion targets are returned, never computed replacement labels.
"""
import ast
from dataclasses import dataclass
import re

from .authored_code_assertions import Unsupported, literal
from .natural_code_context import node_span


def atom(node):
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, bool, type(None)):
        if isinstance(node.value, str) and len(node.value) > 1024:
            raise Unsupported('string_bound')
        if type(node.value) is int and abs(node.value) > 10**9:
            raise Unsupported('integer_bound')
        return node.value
    raise Unsupported('literal_atom_required')


def list_literal(node):
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) > 64:
        raise Unsupported('bounded_literal_sequence_required')
    return [atom(x) for x in node.elts]


@dataclass
class ListState:
    values: list
    start: int
    changes: int = 0


def verified_container_assertions(source):
    if len(source) > 100000:
        raise Unsupported('source_length')
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 12000:
        raise Unsupported('AST_size')
    if any((isinstance(n, ast.alias) and (n.asname or n.name) in ('len', '*'))
           or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == 'len')
           or (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and n.attr == 'len')
           or (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == 'len')
           or (isinstance(n, ast.arg) and n.arg == 'len') for n in nodes):
        return []
    rows = []

    def scope(statements):
        env = {}
        # Explicit len rebinding is rejected for the entire source above.
        # This does not prove runtime absence of arbitrary monkey-patching.
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                        a.arg == 'len' for a in [*statement.args.posonlyargs, *statement.args.args, *statement.args.kwonlyargs]):
                    env.clear()
                    continue
                scope(statement.body)
                env.clear()
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and type(statement.value.value) is str:
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                try:
                    if not isinstance(statement.value, ast.List):
                        raise Unsupported('fresh_list_only_no_alias')
                    env[statement.targets[0].id] = ListState(list_literal(statement.value), node_span(source, statement)[0])
                except Unsupported:
                    env.clear()
                continue
            call = statement.value if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) else None
            if call is not None and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id in env:
                state = env[call.func.value.id]
                try:
                    if call.keywords or any(isinstance(a, ast.Starred) for a in call.args):
                        raise Unsupported('ordinary_positional_arguments_only')
                    before = len(state.values)
                    if call.func.attr == 'append' and len(call.args) == 1:
                        state.values.append(atom(call.args[0]))
                    elif call.func.attr == 'extend' and len(call.args) == 1:
                        state.values.extend(list_literal(call.args[0]))
                    elif call.func.attr == 'pop' and len(call.args) <= 1:
                        index = literal(call.args[0]) if call.args else -1
                        state.values.pop(index)
                    else:
                        raise Unsupported('unsupported_mutation')
                    state.changes += int(len(state.values) != before)
                    if len(state.values) > 64 or state.changes > 12:
                        raise Unsupported('state_bound')
                except (Unsupported, IndexError):
                    env.clear()
                continue
            comparison = statement.test if isinstance(statement, ast.Assert) else None
            # assertEqual is counted in the census but deliberately excluded:
            # its method semantics need not be standard unittest semantics.
            if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
                env.clear()
                continue
            lhs, rhs = comparison.left, comparison.comparators[0]
            if not (isinstance(lhs, ast.Call) and isinstance(lhs.func, ast.Name) and lhs.func.id == 'len'
                    and len(lhs.args) == 1 and not lhs.keywords and isinstance(lhs.args[0], ast.Name) and lhs.args[0].id in env):
                env.clear()
                continue
            state = env[lhs.args[0].id]
            try:
                expected = literal(rhs)
            except Unsupported:
                env.clear()
                continue
            if expected != len(state.values):
                break
            start, end = node_span(source, rhs)
            if state.changes >= 2 and expected >= 2 and not re.search(
                    r'(?<![\w.])'+re.escape(str(expected))+r'(?![\w.])', source[state.start:start]):
                rows.append(dict(target_start=start, target_end=end, premise_start=state.start,
                                 premise_end=start, operations=state.changes, dependency_names=1))
    scope(tree.body)
    return rows
