"""Source-authored single-operation prerequisites; never manufacture answers."""
import ast
import hashlib
import re

from .authored_direct_arithmetic import RUN, SCALAR
from .direct_arithmetic_panel import expression_key
from .natural_code_context import token_span
from .source_arithmetic import check, normalize, Unsupported, PREFIX, FUNCTIONS

OPERATORS = {'Add':'add', 'Sub':'subtract', 'Mult':'multiply', 'Div':'divide'}


def equalities(text):
    if len(text) > 100000:
        raise Unsupported('document_length')
    ends = {m.end():m.start() for m in RUN.finditer(text)}
    for equal in re.finditer('=', text):
        a = ends.get(equal.start())
        if a is None:
            continue
        b = equal.start()
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b-1].isspace(): b -= 1
        if a == b or text[a] in '+-−':
            continue
        before = text[:a].rstrip()
        word = re.search(r'[A-Za-z]+$', before)
        if word and (len(word.group()) == 1 or word.group().lower() in FUNCTIONS
                     or (a > 0 and not text[a-1].isspace())):
            continue
        if before.endswith(('=', '<', '>', '!', '\\', '_')):
            continue
        try:
            premise = check(text[a:b])
            tree = ast.parse(normalize(text[a:b]), mode='eval').body
        except (Unsupported, ValueError, OverflowError, RecursionError, SyntaxError):
            continue
        if (premise.operations != 1 or not isinstance(tree, ast.BinOp)
                or type(tree.op).__name__ not in OPERATORS
                or any(v.denominator != 1 or abs(v) > 1000 for v in premise.literals)
                or premise.value.denominator != 1 or abs(premise.value) > 100000):
            continue
        start = equal.end()
        while start < len(text) and (text[start].isspace() or text[start] == '$'):
            start += 1
        raw = PREFIX.match(text, start)
        if raw is None:
            continue
        end = raw.end()
        while end > start and text[end-1].isspace(): end -= 1
        if end > start and text[end-1] == '.':
            end -= 1
        fragment = text[start:end]
        if not SCALAR.fullmatch(fragment) or '/' in fragment or '.' in fragment:
            continue
        if end < len(text) and (text[end].isalnum() or text[end] in ('_', '\\')):
            continue
        try:
            answer = check(fragment)
        except (Unsupported, ValueError, OverflowError, RecursionError):
            continue
        if answer.value != premise.value or answer.value in premise.literals:
            continue
        yield dict(premise_start=a, premise_end=b, target_start=start, target_end=end,
                   operations=1, stratum=OPERATORS[type(tree.op).__name__])


def eligible_cases(text, encoding):
    for row in equalities(text):
        target = token_span(text, encoding.offsets, row['target_start'], row['target_end'])
        premise = token_span(text, encoding.offsets, row['premise_start'], row['premise_end'])
        if target is None or premise is None:
            continue
        a, b = target
        start = max(0, b-257)
        if a <= start or b-a > 16 or premise[0] < start:
            continue
        answer = text[row['target_start']:row['target_end']]
        prior = text[encoding.offsets[start][0]:row['premise_start']]
        if re.search(r'(?<![\w.])'+re.escape(answer)+r'(?!\w|\.\d)', prior):
            continue
        yield dict(**row, target_tokens=[a,b], premise_tokens=list(premise), window_start=start,
                   expression_key=expression_key(text[row['premise_start']:row['premise_end']]))


def choose_panel(candidates, quota=64, minimum=32):
    selected, groups, expressions, counts = [], set(), set(), {}
    for label in ('divide', 'subtract', 'multiply', 'add'):
        rows = sorted((r for r in candidates if r['stratum'] == label),
                      key=lambda r:hashlib.sha256((r['document_id']+':'+str(r['target_tokens'][0])+':primitive-v1').encode()).hexdigest())
        own = []
        for row in rows:
            if row['group'] in groups or row['expression_key'] in expressions:
                continue
            own.append(row); groups.add(row['group']); expressions.add(row['expression_key'])
            if len(own) == quota:
                break
        selected.extend(own); counts[label] = len(own)
    return selected, counts, all(v >= minimum for v in counts.values())
