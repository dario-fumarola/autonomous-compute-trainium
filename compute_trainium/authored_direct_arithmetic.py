"""Find verified multi-operation equalities already written in natural text.

Return original source spans only. Arithmetic values validate authored answers;
they never supply a replacement answer, trace or transformed training string.
"""
import re

from .source_arithmetic import check, Unsupported, ATOM, PREFIX, FUNCTIONS

RUN = re.compile(ATOM + r'+')
SCALAR = re.compile(r'[+\-−]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*\d+)?')


def direct_equalities(text):
    if len(text) > 100000:
        raise Unsupported('document_length')
    ends = {m.end(): m.start() for m in RUN.finditer(text)}
    for equal in re.finditer('=', text):
        a = ends.get(equal.start())
        if a is None:
            continue
        b = equal.start()
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b-1].isspace(): b -= 1
        before = text[:a].rstrip()
        word = re.search(r'[A-Za-z]+$', before)
        # Avoid treating the numeric tail of an unknown symbolic expression as
        # a complete premise (including implicit products and function calls).
        if word and (len(word.group()) == 1 or word.group().lower() in FUNCTIONS
                     or (a > 0 and not text[a-1].isspace())):
            continue
        if before.endswith(('=', '<', '>', '!', '\\', '_')):
            continue
        try:
            premise = check(text[a:b])
        except (Unsupported, ValueError, OverflowError, RecursionError):
            continue
        if premise.operations < 2:
            continue
        start = equal.end()
        while start < len(text) and (text[start].isspace() or text[start] == '$'):
            start += 1
        raw = PREFIX.match(text, start)
        if raw is None:
            continue
        end = raw.end()
        while end > start and text[end-1].isspace(): end -= 1
        fragment = text[start:end]
        if not SCALAR.fullmatch(fragment):
            continue
        if end < len(text) and (text[end].isalnum() or text[end] in ('_', '\\')):
            continue
        try:
            answer = check(fragment)
        except (Unsupported, ValueError, OverflowError, RecursionError):
            continue
        if answer.value != premise.value or answer.value in premise.literals:
            continue
        yield dict(premise_start=a, premise_end=b, target_start=start,
                   target_end=end, operations=premise.operations)
