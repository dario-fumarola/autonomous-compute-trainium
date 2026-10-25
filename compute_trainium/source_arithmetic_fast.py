"""Same source-chain rules,with one maximal-token scan per document.

The arithmetic checker and source-span semantics are unchanged. A complete
same-cohort equivalence gate is required before larger source reconstruction.
"""
import re
from .source_arithmetic import check,Unsupported,ATOM,PREFIX,FUNCTIONS
RUN=re.compile(ATOM+r'+')

def original_chains(text):
    """Yield source spans only. All arithmetic values are validation-local."""
    runs={m.end():m.start() for m in RUN.finditer(text)}
    for match in re.finditer('=',text):
        lhs_end=match.start()
        if lhs_end not in runs:continue
        a=runs[lhs_end];b=lhs_end
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        before=text[:a].rstrip();word=re.search(r'[A-Za-z]+$',before)
        if word and (len(word.group())==1 or word.group().lower() in FUNCTIONS or (a>0 and not text[a-1].isspace())):continue
        try:first=check(text[a:b])
        except (Unsupported,ValueError,OverflowError):continue
        if first.operations<2:continue
        spans=[[a,b]];values=[first];cursor=match.end()
        for _ in range(5):
            while cursor<len(text) and (text[cursor].isspace() or text[cursor]=='$'):cursor+=1
            rhs=PREFIX.match(text,cursor)
            if not rhs:break
            end=rhs.end()
            while end>cursor and text[end-1].isspace():end-=1
            try:parsed=check(text[cursor:end])
            except (Unsupported,ValueError,OverflowError):break
            if parsed.value!=first.value:break
            spans.append([cursor,end]);values.append(parsed);cursor=rhs.end()
            while cursor<len(text) and (text[cursor].isspace() or text[cursor]=='$'):cursor+=1
            scalar=parsed.operations==0 or bool(re.fullmatch(r'[+\-−]?\s*\d+(?:\.\d+)?\s*/\s*\d+',text[spans[-1][0]:end]))
            if scalar:
                if len(spans)>=3 and any(0<v.operations<first.operations for v in values[1:-1]) and first.value not in frozenset().union(*(v.literals for v in values[:-1])):
                    yield dict(expression=spans[0],intermediate_steps=spans[1:-1],conclusion=spans[-1],
                        original_operations=first.operations,step_operations=[v.operations for v in values[1:-1]])
                break
            if cursor>=len(text) or text[cursor]!='=':break
            cursor+=1
