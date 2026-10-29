"""TRAIN source-only structural-dependency inventory; no execution or model scores."""
import argparse, ast, bisect, collections, hashlib, io, json, resource, time, tokenize, warnings
from pathlib import Path

PAIRS={')':'(',']':'[','}':'{'}
def closing_spans(text):
    """Return authored closers and their lexical opener, excluding strings/comments."""
    lines=text.splitlines(keepends=True);starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    stack=[];rows=[]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type!=tokenize.OP:continue
        value=token.string
        char=starts[token.start[0]-1]+token.start[1]
        if value in PAIRS.values():stack.append((value,char,token.start[0]))
        elif value in PAIRS:
            if not stack or stack[-1][0]!=PAIRS[value]:raise ValueError('Unbalanced lexical source')
            depth=len(stack);opener,begin,line=stack.pop()
            rows.append(dict(begin=begin,end=char,depth=depth,line_distance=token.start[0]-line,kind=value))
    if stack:raise ValueError('Incomplete source')
    return rows

def tests():
    # Software fixtures only. None enters a research corpus.
    s='x = [f(1, {"a": [2]})] # ) ] }\ny = "[("\n'
    r=closing_spans(s)
    assert [x['kind'] for x in r]==[']','}',')',']']
    assert [x['depth'] for x in r]==[4,3,2,1]
    assert all(s[x['begin']]==PAIRS[x['kind']] and s[x['end']]==x['kind'] for x in r)
    s='é = (\n  [1, 2]\n)\n';r=closing_spans(s)
    assert r[1]['line_distance']==2 and r[1]['begin']==4
    assert closing_spans('x=""" ) } [ """\n')==[]
    try:closing_spans('x = ([)]\n')
    except ValueError:pass
    else:raise AssertionError('Mismatch accepted')
    return 4

def main(a):
    import numpy as np
    from tokenizers import Tokenizer
    from compute_trainium.training import file_sha256
    started=time.time();fixtures=tests();data=a.root/'probability-workspace-data-v1';receipt=json.loads((data/'report.json').read_text())
    for name in ('PRIVATE-training-documents.jsonl','train-tokens.npy'):
        if file_sha256(data/name)!=receipt['files'][name]:raise AssertionError('Frozen TRAIN source differs')
    tk=Tokenizer.from_file(str(a.tokenizer));tokens=np.load(data/'train-tokens.npy',mmap_mode='r');counts=collections.Counter();bands=collections.Counter();groups=set();eligible_groups=set()
    for raw in (data/'PRIVATE-training-documents.jsonl').open():
        row=json.loads(raw)
        if row['start']>=200_000_000:break
        if row['domain']!='code':continue
        if counts['documents_read']>=10000:break
        counts['documents_read']+=1;groups.add(row['group']);n=row['source_tokens'];ids=tokens[row['start']+1:row['start']+1+n].tolist();counts['source_tokens_read']+=n
        text=tk.decode(ids,skip_special_tokens=False);enc=tk.encode(text,add_special_tokens=False)
        if enc.ids!=ids:counts['roundtrip_rejected']+=1;continue
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');ast.parse(text)
            counts['parse_warnings']+=len(ws);rows=closing_spans(text)
        except (SyntaxError,ValueError,tokenize.TokenError,IndentationError,RecursionError):counts['parse_rejected_unknown']+=1;continue
        counts['parse_accepted']+=1;ends=[b for _,b in enc.offsets];document_eligible=False
        for r in rows:
            counts['authored_closers']+=1
            left=bisect.bisect_right(ends,r['begin']);target=bisect.bisect_right(ends,r['end'])
            if left>=len(ends) or target>=len(ends):raise AssertionError('Lexical/token alignment outside source')
            distance=target-left;a0,b0=enc.offsets[target]
            if not (a0<=r['end']<b0):raise AssertionError('Target token misses authored closer')
            strict=not text[a0:r['end']].strip() and not text[r['end']+1:b0].strip()
            band=('0-15' if distance<16 else '16-63' if distance<64 else '64-255' if distance<256 else '256-511' if distance<512 else '512+')
            depth=str(min(r['depth'],5));bands['all|'+band+'|depth'+depth]+=1
            if not strict:counts['closer_shares_token_with_nonwhitespace']+=1;continue
            counts['strict_authored_target_tokens']+=1;bands['strict|'+band+'|depth'+depth]+=1
            if 64<=distance<512:
                counts['strict_distance64_511']+=1
                if r['depth']>=2:
                    counts['strict_distance64_511_depth2plus']+=1;document_eligible=True
                    if r['line_distance']>0:counts['strict_distance64_511_depth2plus_multiline']+=1
        if document_eligible:counts['documents_strict_distance64_511_depth2plus']+=1;eligible_groups.add(row['group'])
        if counts['documents_read']%128==0 and (time.time()-started>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3):raise RuntimeError('Inventory resource bound exceeded')
    result=dict(status='COMPLETE_TRAIN_CODE_STACK_INVENTORY',counts=dict(counts),distance_depth_histogram=dict(sorted(bands.items())),source_groups=len(groups),eligible_source_groups=len(eligible_groups),software_fixture_checks=fixtures,
      script_sha256=file_sha256(__file__),data_report_sha256=file_sha256(data/'report.json'),tokenizer_sha256=file_sha256(a.tokenizer),wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,assigned_chips=0,optimizer_updates=0,heldout_reads=0,
      scope='First10000 TRAIN code documents in the first200M packed positions,unchanged frozen population. Only AST/tokenizer parsing;no source execution,no calculated labels,no model reads. Counts authentic closing-delimiter targets with causal lexical opener and original intervening context. Strict targets contain only the authored closer plus whitespace. This tests availability of structural dependencies,not multistep reasoning or an evaluation result. Syntax/completion shortcuts remain possible;openers inside context do not prove a model uses them. A prospective panel and source interventions would require a separate registration. Current training,evaluation and gates remain unchanged. Parser-rejected/truncated inputs are unknown.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='distance_depth_histogram'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
