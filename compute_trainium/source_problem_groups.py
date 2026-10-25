"""Conservative duplicate grouping for natural arithmetic validation."""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from .source_arithmetic import normalize,check


def literal_operator_group(expression):
    check(expression);text=normalize(expression);tree=ast.parse(text,mode='eval')
    literals=Counter();operators=Counter()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant):literals[str(Fraction(ast.get_source_segment(text,node)))]+=1
        if isinstance(node,(ast.BinOp,ast.UnaryOp)):operators[type(node.op).__name__]+=1
    # Intentionally merges distinct parenthesizations/orderings of the same
    # values/operators. It is an exclusion key,not an equivalence theorem.
    return hashlib.sha256(json.dumps([sorted(literals.items()),sorted(operators.items())],separators=(',',':')).encode()).hexdigest()
