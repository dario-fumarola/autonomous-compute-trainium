"""Verify literal results already present in authentic Python docstrings.

Only a bounded, pure integer subset is interpreted. Never execute source code,
invent a target, repair an example or turn nonmatches into research examples.
"""
import ast
import re

from .authored_code_assertions import Unsupported, expression, bounded
from .natural_code_context import node_span

PROMPT = re.compile(r'^[ \t]*>>>[ \t]+(.+?)[ \t]*(?:\r?\n)?$')
ANSWER = re.compile(r'^[ \t]*(-?(?:0|[1-9][0-9]*))[ \t]*(?:\r?\n)?$')
OPEN = re.compile(r"(?i)^(?:r|u)?('''|\"\"\")")


def verified_doctest_arithmetic(source):
    if len(source)>100000:raise Unsupported('source_length')
    tree=ast.parse(source)
    if sum(1 for _ in ast.walk(tree))>12000:raise Unsupported('AST_size')
    rows=[]
    for scope in ast.walk(tree):
        if not isinstance(scope,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) or not scope.body:
            continue
        first=scope.body[0]
        if not (isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and type(first.value.value) is str):
            continue
        lo,hi=node_span(source,first.value);raw=source[lo:hi];opening=OPEN.match(raw)
        if not opening or not raw.endswith(opening[1]):continue
        body_start=lo+opening.end();body=source[body_start:hi-3]
        lines=body.splitlines(keepends=True);starts=[];offset=body_start
        for line in lines:starts.append(offset);offset+=len(line)
        values,origins={},{}
        for i,line in enumerate(lines):
            prompt=PROMPT.fullmatch(line)
            if not prompt:continue
            try:
                parsed=ast.parse(prompt[1]);statements=parsed.body
                if len(statements)!=1:raise Unsupported('one_statement_only')
                statement=statements[0]
                if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
                    node=statement.value;assignment=statement.targets[0].id
                elif isinstance(statement,ast.Expr):node=statement.value;assignment=None
                else:raise Unsupported('pure_assignment_or_expression_only')
                value=expression(node,values)
                names=[n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)]
                premise=min([starts[i]]+[origins[n] for n in names])
                if assignment is not None:
                    values[assignment]=value;origins[assignment]=premise;continue
                if i+1>=len(lines):continue
                answer=ANSWER.fullmatch(lines[i+1])
                if answer is None:continue
                expected=bounded(int(answer[1]))
                if answer[1]!=str(value.integer) or expected!=value.integer:
                    values.clear();origins.clear();continue
                target_start=starts[i+1]+answer.start(1);target_end=starts[i+1]+answer.end(1)
                if value.operations<2 or expected in value.literals:continue
                if re.search(r'(?<![\w.])'+re.escape(str(expected))+r'(?![\w.])',source[premise:target_start]):continue
                rows.append(dict(target_start=target_start,target_end=target_end,
                    premise_start=premise,premise_end=target_start,operations=value.operations,
                    dependency_names=len(names)))
            except (Unsupported,SyntaxError,ValueError,RecursionError):
                values.clear();origins.clear()
    return rows
