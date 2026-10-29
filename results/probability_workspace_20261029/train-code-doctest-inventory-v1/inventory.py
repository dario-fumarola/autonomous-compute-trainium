"""Syntax-only inventory of authentic TRAIN function-call assertions."""
import argparse,ast,collections,hashlib,json,resource,time,warnings,doctest
from pathlib import Path

def literal(node):
    # Classification only: do not evaluate source expressions or execute code.
    if isinstance(node,ast.Constant):return type(node.value) in (int,float,bool,str,type(None))
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):return isinstance(node.operand,ast.Constant) and type(node.operand.value) in (int,float)
    if isinstance(node,(ast.Tuple,ast.List)):return len(node.elts)<=32 and all(literal(x) for x in node.elts)
    return False

def candidates(tree):
    definitions=collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):definitions[node.name].append(node)
    for node in ast.walk(tree):
        left=right=None
        if isinstance(node,ast.Assert) and isinstance(node.test,ast.Compare) and len(node.test.ops)==1 and isinstance(node.test.ops[0],ast.Eq):
            left,right=node.test.left,node.test.comparators[0]
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('assertEqual','assertEquals') and len(node.args)==2:
            left,right=node.args
        if not isinstance(left,ast.Call) or not literal(right):continue
        direct=isinstance(left.func,ast.Name);name=left.func.id if direct else None
        defs=definitions.get(name,[]);definition=defs[0] if len(defs)==1 and defs[0].end_lineno<node.lineno else None
        yield dict(assertion=node,call=left,target=right,definition=definition,
                   direct=direct,literal_arguments=all(literal(x) for x in left.args) and all(x.arg is not None and literal(x.value) for x in left.keywords))

def main(a):
    import numpy as np
    from tokenizers import Tokenizer
    from compute_trainium.natural_code_context import node_span,token_span
    from compute_trainium.training import file_sha256
    start=time.time();data=a.root/'probability-workspace-data-v1';receipt=json.loads((data/'report.json').read_text())
    if file_sha256(data/'PRIVATE-training-documents.jsonl')!=receipt['files']['PRIVATE-training-documents.jsonl']:raise AssertionError('Frozen training recipe changed')
    if file_sha256(data/'train-tokens.npy')!=receipt['files']['train-tokens.npy']:raise AssertionError('Frozen training token bytes changed')
    tokenizer=Tokenizer.from_file(str(a.tokenizer));tokens=np.load(data/'train-tokens.npy',mmap_mode='r');counts=collections.Counter();groups=set();context_counts=collections.Counter();operations=collections.Counter()
    for line in (data/'PRIVATE-training-documents.jsonl').open():
        row=json.loads(line)
        if row['start']>=200_000_000:break
        if row['domain']!='code':continue
        if counts['documents_read']>=10000:break
        counts['documents_read']+=1;groups.add(row['group']);n=row['source_tokens'];ids=tokens[row['start']+1:row['start']+1+n].tolist();counts['source_tokens_read']+=n
        text=tokenizer.decode(ids,skip_special_tokens=False)
        encoded=tokenizer.encode(text,add_special_tokens=False)
        if encoded.ids!=ids:counts['token_roundtrip_rejected']+=1;continue
        counts['token_roundtrip_accepted']+=1
        try:
            with warnings.catch_warnings(record=True) as parse_warnings:
                warnings.simplefilter('always');tree=ast.parse(text)
            counts['parse_warnings']+=len(parse_warnings)
        except (SyntaxError,ValueError,RecursionError,MemoryError):counts['ast_parse_rejected']+=1;continue
        nodes=list(ast.walk(tree))
        if len(nodes)>50000:counts['ast_size_rejected']+=1;continue
        counts['ast_parse_accepted']+=1;found=list(candidates(tree))
        # Count source-authored doctest opportunities only; never run examples.
        for fn in nodes:
            if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            doc=ast.get_docstring(fn,clean=False)
            if not doc:continue
            counts['functions_with_docstrings']+=1
            try:examples=doctest.DocTestParser().get_examples(doc)
            except ValueError:counts['doctest_parse_rejected']+=1;continue
            bodyops=sum(isinstance(z,(ast.BinOp,ast.UnaryOp,ast.Compare)) for z in ast.walk(fn))
            for ex in examples:
                if not ex.want.strip():continue
                counts['doctest_examples_with_authored_output']+=1
                try:
                    with warnings.catch_warnings(record=True):
                        inp=ast.parse(ex.source);expected=ast.parse(ex.want.strip(),mode='eval').body
                except (SyntaxError,ValueError,RecursionError):counts['doctest_nonliteral_or_multiline_output']+=1;continue
                if not literal(expected):continue
                counts['doctest_literal_outputs']+=1
                if len(inp.body)!=1 or not isinstance(inp.body[0],ast.Expr):continue
                call=inp.body[0].value
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!=fn.name:continue
                counts['doctest_calls_own_function']+=1
                if not all(literal(v) for v in call.args) or not all(v.arg is not None and literal(v.value) for v in call.keywords):continue
                counts['doctest_own_function_literal_arguments']+=1
                if bodyops>=2:counts['doctest_own_function_literal_with_two_syntax_operations']+=1

        if found:counts['documents_with_call_literal_assertion']+=1
        for r in found:
            counts['call_literal_assertions']+=1
            if not r['direct']:counts['non_name_callee']+=1
            if r['literal_arguments']:counts['literal_argument_assertions']+=1
            fn=r['definition']
            if fn is None:continue
            counts['prior_unique_top_level_definition']+=1
            if not r['literal_arguments']:continue
            counts['local_definition_and_literal_arguments']+=1
            body=list(ast.walk(fn));ops=sum(isinstance(z,(ast.BinOp,ast.UnaryOp,ast.Compare)) for z in body)
            operations[str(min(ops,10))]+=1
            if ops<2:continue
            counts['local_literal_with_at_least_two_syntax_operations']+=1
            begin,_=node_span(text,fn);lo,hi=node_span(text,r['target']);span=token_span(text,encoded.offsets,lo,hi)
            if span is None:counts['target_token_alignment_rejected']+=1;continue
            start_tokens=[i for i,(_,b) in enumerate(encoded.offsets) if b>begin]
            if not start_tokens:counts['definition_token_alignment_rejected']+=1;continue
            length=span[1]-start_tokens[0]
            for maximum in (512,1024,2048):
                if length<=maximum:context_counts[str(maximum)]+=1
        if counts['documents_read']%128==0:
            if time.time()-start>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise RuntimeError('CPU inventory bound exceeded')
    result=dict(status='COMPLETE_TRAIN_CODE_ASSERTION_AND_DOCTEST_INVENTORY',counts=dict(counts),source_groups=len(groups),context_fit_counts=dict(context_counts),operation_histogram=dict(operations),
      data_report_sha256=file_sha256(data/'report.json'),tokenizer_sha256=file_sha256(a.tokenizer),script_sha256=file_sha256(__file__),
      wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,assigned_chips=0,optimizer_updates=0,heldout_reads=0,
      scope='First at most10000 code documents within first200M packed TRAIN positions,from the frozen source recipe. Source tokens are decoded and exact-roundtrip checked. No source program is executed,no expected answer is calculated,no target or example is generated,no model score is read. Syntax opportunities are not verified correct assertions or proof of required serial computation. Rejected/truncated/unparseable files are unknown,not negative examples. Context counts are nested and include original intervening source. Doctest outputs are source-authored but commonly precede the function body in the source;they are prospective task opportunities,not valid causal-prefix examples as-is. Current fitting/evaluation rules are unchanged;no new fit or panel admitted.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
