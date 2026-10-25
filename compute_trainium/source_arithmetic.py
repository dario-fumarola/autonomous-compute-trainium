"""Bounded verification of existing source expressions;never creates labels."""
import ast
from dataclasses import dataclass
from fractions import Fraction
import re


class Unsupported(ValueError):pass


@dataclass(frozen=True)
class Checked:
    value:Fraction
    operations:int
    literals:frozenset


def normalize(expression):
    if not expression or len(expression)>240:raise Unsupported('expression_length')
    text=expression.replace('−','-').replace('×','*').replace('÷','/')
    text=text.replace('\\dfrac','\\frac').replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
    text=text.replace('\\left','').replace('\\right','').strip()
    def group(start):
        if start>=len(text) or text[start]!='{':raise Unsupported('fraction_group')
        level=1;end=start+1
        while end<len(text) and level:
            level+=(text[end]=='{')-(text[end]=='}');end+=1
        if level:raise Unsupported('unbalanced_group')
        return text[start+1:end-1],end
    # Innermost recursion is bounded by the original240-character input.
    if '\\frac' in text:
        start=text.index('\\frac');a=start+5
        while a<len(text) and text[a].isspace():a+=1
        numerator,b=group(a)
        while b<len(text) and text[b].isspace():b+=1
        denominator,end=group(b)
        converted='('+normalize(numerator)+')/('+normalize(denominator)+')'
        return normalize(text[:start]+'('+converted+')'+text[end:])
    text=text.replace('{','(').replace('}',')').replace('^','**')
    text=re.sub(r'\s+',' ',text)
    if re.search(r'[^\d\s.()+*/-]',text):raise Unsupported('non_numeric_input')
    return text


def check(expression):
    text=normalize(expression)
    try:tree=ast.parse(text,mode='eval')
    except (SyntaxError,ValueError,RecursionError) as error:raise Unsupported('invalid_AST') from error
    if sum(1 for _ in ast.walk(tree))>63:raise Unsupported('node_bound')
    operations=0;literals=set()
    def bounded(x):
        if abs(x)>10**12 or abs(x.numerator)>10**18 or x.denominator>10**18:raise Unsupported('value_bound')
        return x
    def walk(node):
        nonlocal operations
        if isinstance(node,ast.Constant) and type(node.value) in [int,float]:
            literal=ast.get_source_segment(text,node)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',literal or ''):raise Unsupported('numeric_literal')
            value=bounded(Fraction(literal));literals.add(value);return value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=walk(node.operand);value=-value if isinstance(node.op,ast.USub) else value
            if isinstance(node.operand,ast.Constant):literals.add(value)
            return value
        if not isinstance(node,ast.BinOp) or not isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            raise Unsupported('non_arithmetic_AST')
        operations+=1
        if operations>6:raise Unsupported('operation_bound')
        a,b=walk(node.left),walk(node.right)
        try:
            if isinstance(node.op,ast.Add):value=a+b
            elif isinstance(node.op,ast.Sub):value=a-b
            elif isinstance(node.op,ast.Mult):value=a*b
            elif isinstance(node.op,ast.Div):value=a/b
            else:
                if b.denominator!=1 or abs(b)>8:raise Unsupported('power_bound')
                value=a**int(b)
        except ZeroDivisionError as error:raise Unsupported('division_by_zero') from error
        return bounded(value)
    value=walk(tree.body)
    return Checked(value,operations,frozenset(literals))


ATOM=r'(?:\\(?:dfrac|frac|times|cdot|div|left|right)|[\d\s.()+*/^{}\-−×÷])'
SUFFIX=re.compile(ATOM+r'+$')
PREFIX=re.compile(ATOM+r'+')
FUNCTIONS={'sin','cos','tan','log','ln','exp','sqrt','abs','min','max','floor','ceil','round'}


def original_chains(text):
    """Yield source spans only. All arithmetic values are validation-local."""
    for match in re.finditer('=',text):
        lhs_end=match.start();suffix=SUFFIX.search(text[:lhs_end])
        if not suffix:continue
        a=suffix.start();b=lhs_end
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
