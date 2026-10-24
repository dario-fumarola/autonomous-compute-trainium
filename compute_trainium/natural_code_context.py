"""Syntax-only natural identifier selection; never executes source programs."""
import ast
from collections import Counter


def same_scope(function):
    def visit(node):
        yield node
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda,
                            ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)):
            return
        for child in ast.iter_child_nodes(node):yield from visit(child)
    for statement in function.body:yield from visit(statement)


def character_position(source,line,column):
    lines=source.splitlines(keepends=True)
    if not 1<=line<=len(lines):raise ValueError('Source line outside file')
    prefix=lines[line-1].encode('utf8')[:column].decode('utf8')
    return sum(map(len,lines[:line-1]))+len(prefix)


def node_span(source,node):
    return (character_position(source,node.lineno,node.col_offset),
            character_position(source,node.end_lineno,node.end_col_offset))


def unambiguous_local_uses(source):
    """One direct unconditional binder, no other binding in that local scope.

    Later conditional uses are allowed when the direct body assignment dominates
    them. Nested scopes are excluded; indirect/global binding constructs reject
    the whole function. No values, types or model scores enter selection.
    """
    tree=ast.parse(source);rows=[]
    for function in ast.walk(tree):
        if not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        nodes=list(same_scope(function))
        if any(isinstance(n,(ast.Global,ast.Nonlocal,ast.Import,ast.ImportFrom,ast.Match)) for n in nodes):continue
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'exec','eval','globals','locals'} for n in nodes):continue
        stores=Counter(n.id for n in nodes if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Del)))
        for arg in [*function.args.posonlyargs,*function.args.args,*function.args.kwonlyargs,
                    *([function.args.vararg] if function.args.vararg else []),
                    *([function.args.kwarg] if function.args.kwarg else [])]:stores[arg.arg]+=1
        for n in nodes:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):stores[n.name]+=1
            if isinstance(n,ast.ExceptHandler) and n.name:stores[n.name]+=1
        definitions={}
        for statement in function.body:
            name=(statement.targets[0] if isinstance(statement,ast.Assign) and len(statement.targets)==1 else
                  statement.target if isinstance(statement,ast.AnnAssign) and statement.value is not None else None)
            if isinstance(name,ast.Name) and stores[name.id]==1:
                definitions[name.id]=(name,statement)
        for use in nodes:
            if not isinstance(use,ast.Name) or not isinstance(use.ctx,ast.Load) or use.id not in definitions:continue
            name,definition=definitions[use.id];start,end=node_span(source,use)
            dstart,dend=node_span(source,name);_,statement_end=node_span(source,definition)
            if statement_end>start:continue
            if source[start:end]!=use.id or source[dstart:dend]!=use.id:raise AssertionError('UTF8 AST identifier alignment differs')
            rows.append({'function_line':function.lineno,'definition_start':dstart,'definition_end':dend,
                         'definition_statement_end':statement_end,'target_start':start,'target_end':end})
    return rows


def token_span(source,offsets,start,end):
    touched=[i for i,(a,b) in enumerate(offsets) if b>start and a<end]
    if not touched:return None
    lo,hi=touched[0],touched[-1]+1;a=offsets[lo][0];b=offsets[hi-1][1]
    if a>start or b<end or source[a:start].strip() or source[end:b].strip():return None
    return lo,hi


def distance_band(distance):
    for low,high in [(129,256),(257,512),(513,900)]:
        if low<=distance<=high:return f'{low}-{high}'
    return None
