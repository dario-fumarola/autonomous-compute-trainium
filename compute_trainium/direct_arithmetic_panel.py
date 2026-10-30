"""Deterministic source-only selection for an unassisted arithmetic diagnostic."""
import ast
from fractions import Fraction
import hashlib
import json
import re

from .authored_direct_arithmetic import direct_equalities
from .natural_code_context import token_span
from .source_arithmetic import normalize


def expression_key(expression, *, blind=False):
    text=normalize(expression)
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            return ('number','_' if blind else str(Fraction(ast.get_source_segment(text,n))))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return (type(n.op).__name__,visit(n.operand))
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            return (type(n.op).__name__,visit(n.left),visit(n.right))
        raise ValueError('Verified arithmetic AST only')
    return hashlib.sha256(json.dumps(visit(ast.parse(text,mode='eval').body),separators=(',',':')).encode()).hexdigest()


def eligible_cases(text, encoding):
    """At most256 original tokens including answer; no added framing or work."""
    for row in direct_equalities(text):
        # A sentence-final period need not become an arithmetic answer target.
        # The retained answer remains an exact original substring.
        if text[row['target_end']-1]=='.':
            row={**row,'target_end':row['target_end']-1}
        expression=text[row['premise_start']:row['premise_end']]
        # A leading binary-looking sign can be the tail of a longer symbolic
        # expression separated by whitespace. Conservatively omit it, including
        # legitimate prose followed by an unparenthesized negative first operand.
        if expression[0] in '+-−': continue
        target=token_span(text,encoding.offsets,row['target_start'],row['target_end'])
        premise=token_span(text,encoding.offsets,row['premise_start'],row['premise_end'])
        if target is None or premise is None: continue
        a,b=target; start=max(0,b-257)
        if a<=start or b-a>16 or premise[0]<start: continue
        answer=text[row['target_start']:row['target_end']]
        prior=text[encoding.offsets[start][0]:row['premise_start']]
        # The complete authored answer must not already be visible as the same
        # numeric string in this context. This does not prove semantic novelty.
        if re.search(r'(?<![\w.])'+re.escape(answer)+r'(?!\w|\.\d)',prior): continue
        yield dict(**row,target_tokens=[a,b],premise_tokens=list(premise),window_start=start,
                   expression_key=expression_key(expression),shape_key=expression_key(expression,blind=True),
                   stratum='2' if row['operations']==2 else '3' if row['operations']==3 else '4-6')


def choose_panel(candidates, quota=128, minimum=32):
    """One source group and one literal-valued AST per panel; hard bins first."""
    used_groups=set();used_expressions=set();selected=[];counts={}
    for stratum in ('4-6','3','2'):
        values=[r for r in candidates if r['stratum']==stratum]
        values.sort(key=lambda r:hashlib.sha256((r['document_id']+':'+str(r['target_tokens'][0])+':direct-v1').encode()).hexdigest())
        own=[]
        for row in values:
            if row['group'] in used_groups or row['expression_key'] in used_expressions: continue
            own.append(row);used_groups.add(row['group']);used_expressions.add(row['expression_key'])
            if len(own)==quota: break
        counts[stratum]=len(own);selected.extend(own)
    return selected,counts,all(n>=minimum for n in counts.values())
