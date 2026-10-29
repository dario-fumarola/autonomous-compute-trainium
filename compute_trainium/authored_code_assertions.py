"""Verify existing integer-result assertions without executing source code.

Only straight-line, local integer arithmetic is interpreted. Returned labels
are original source spans; computed values are validation-local and never
returned as replacement targets. Nonmatches are not negative task examples.
"""
import ast
from dataclasses import dataclass

from .natural_code_context import node_span


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Value:
    integer: int
    operations: int
    literals: frozenset


def bounded(value):
    if type(value) is not int or abs(value) > 10**9:
        raise Unsupported('integer_bound')
    return value


def literal(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return bounded(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        n = literal(node.operand)
        return -n if isinstance(node.op, ast.USub) else n
    raise Unsupported('not_integer_literal')


def expression(node, env, depth=0):
    if depth > 16:
        raise Unsupported('expression_depth')
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in env:
        return env[node.id]
    try:
        n = literal(node)
        return Value(n, 0, frozenset({n}))
    except Unsupported:
        pass
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        raise Unsupported('nonlocal_or_unsupported_expression')
    a, b = expression(node.left, env, depth+1), expression(node.right, env, depth+1)
    operations = a.operations + b.operations + 1
    if operations > 12:
        raise Unsupported('operation_bound')
    if isinstance(node.op, ast.Add):
        value = a.integer + b.integer
    elif isinstance(node.op, ast.Sub):
        value = a.integer - b.integer
    elif isinstance(node.op, ast.Mult):
        value = a.integer * b.integer
    elif isinstance(node.op, ast.FloorDiv):
        if not b.integer:
            raise Unsupported('division_by_zero')
        value = a.integer // b.integer
    else:
        if not b.integer:
            raise Unsupported('division_by_zero')
        value = a.integer % b.integer
    return Value(bounded(value), operations, a.literals | b.literals)


def function_value(function, args):
    formal = function.args
    names = [*formal.posonlyargs, *formal.args]
    if (function.decorator_list or function.returns is not None or formal.vararg or formal.kwarg
            or formal.kwonlyargs or formal.defaults or any(a.annotation is not None for a in names)
            or len(names) != len(args) or len(function.body) > 16):
        raise Unsupported('function_interface')
    env = {a.arg: Value(v, 0, frozenset({v})) for a, v in zip(names, args)}
    for index, statement in enumerate(function.body):
        if (index == 0 and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant) and type(statement.value.value) is str):
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            env[statement.targets[0].id] = expression(statement.value, env)
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name) and statement.target.id in env:
            lhs = ast.Name(id=statement.target.id, ctx=ast.Load())
            env[statement.target.id] = expression(ast.BinOp(left=lhs, op=statement.op, right=statement.value), env)
        elif isinstance(statement, ast.Return) and index == len(function.body)-1:
            return expression(statement.value, env)
        else:
            raise Unsupported('not_straight_line_pure_function')
    raise Unsupported('missing_final_return')


def verified_assertions(source):
    """Top-level authored `assert f(literals) == literal` after a local definition.

    Unknown module statements invalidate known functions because they might
    rebind them. Function bodies read only arguments and prior local assignments.
    No imports, calls, attribute access, loops, globals, eval or exec are run.
    """
    if len(source) > 100000:
        raise Unsupported('source_length')
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 12000:
        raise Unsupported('AST_size')
    functions, rows = {}, []
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            # A decorated/defaulted definition may execute arbitrary code and
            # rebind earlier functions. Conservatively forget the whole table.
            args = statement.args
            all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs,
                        *([args.vararg] if args.vararg else []), *([args.kwarg] if args.kwarg else [])]
            if (statement.decorator_list or args.defaults or args.kw_defaults
                    or statement.returns is not None or any(a.annotation for a in all_args)):
                functions.clear()
                continue
            functions[statement.name] = statement
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and type(statement.value.value) is str:
            continue
        if not isinstance(statement, ast.Assert):
            functions.clear()
            continue
        test = statement.test
        if (not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq)
                or len(test.comparators) != 1 or not isinstance(test.left, ast.Call)
                or not isinstance(test.left.func, ast.Name) or test.left.keywords
                or test.left.func.id not in functions or statement.msg is not None):
            functions.clear()
            continue
        function = functions[test.left.func.id]
        try:
            arguments = [literal(a) for a in test.left.args]
            expected = literal(test.comparators[0])
            result = function_value(function, arguments)
        except Unsupported:
            functions.clear()
            continue
        # A failed assertion ends the real program; no later assertion is a
        # verified continuation. Copyable and one-operation cases are excluded
        # from the computation panel, but do not change subsequent bindings.
        if result.integer != expected:
            break
        operand_literals = set(arguments)
        for node in ast.walk(function):
            try:
                operand_literals.add(literal(node))
            except Unsupported:
                pass
        if result.operations < 2 or expected in operand_literals:
            continue
        start, end = node_span(source, test.comparators[0])
        function_start, function_end = node_span(source, function)
        call_start, call_end = node_span(source, test.left)
        rows.append(dict(target_start=start, target_end=end, function_start=function_start,
                         function_end=function_end, call_start=call_start, call_end=call_end,
                         operations=result.operations, arguments=len(arguments)))
    return rows
