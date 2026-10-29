"""Prospective authentic-code paired panel; never execute or generate source."""
import argparse,ast,bisect,collections,hashlib,importlib.util,json,resource,sqlite3,time,warnings
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from compute_trainium.training import file_sha256

SPEC=importlib.util.spec_from_file_location('stack_inventory',Path(__file__).with_name('stack_inventory.py'))
STACK=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(STACK)

def paired_spans(text,offsets,identity):
    ends=[b for _,b in offsets];rows=[]
    for r in STACK.closing_spans(text):
        opener=bisect.bisect_right(ends,r['begin']);target=bisect.bisect_right(ends,r['end'])
        if opener>=len(ends) or target>=len(ends):raise AssertionError('Source bounds')
        a,b=offsets[target]
        if not(a<=r['end']<b):raise AssertionError('Closer alignment')
        if text[a:r['end']].strip() or text[r['end']+1:b].strip():continue
        if target<32 or r['depth']<2:continue
        distance=target-opener
        if not(1<=distance<16 or 64<=distance<512):continue
        rows.append(dict(**r,opener_token=opener,target_token=target,distance=distance,depth_band=min(r['depth'],4)))
    candidates=[]
    for long in rows:
        if long['distance']<64:continue
        short=[r for r in rows if r['distance']<16 and r['kind']==long['kind'] and r['depth_band']==long['depth_band']]
        if not short:continue
        rank=lambda r:hashlib.sha256((identity+':structural-short-9521:'+str(r['end'])).encode()).hexdigest()
        other=min(short,key=rank)
        key=hashlib.sha256((identity+':structural-long-9521:'+str(long['end'])).encode()).hexdigest()
        candidates.append((key,long,other))
    return None if not candidates else min(candidates,key=lambda r:r[0])[1:]

def main(a):
    started=time.time();root=a.root;out=root/'probability-structural-panel-v1';out.mkdir(mode=0o700,exist_ok=False)
    public=Path('results/probability_workspace_20261029/structural-panel-v1');plan=json.loads((public/'plan.json').read_text())
    if file_sha256(__file__)!=plan['script_sha256'] or file_sha256(Path(__file__).with_name('stack_inventory.py'))!=plan['parser_sha256']:raise AssertionError('Registered code changed')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    data=root/'probability-workspace-data-v1';base=root/'natural-append-3250m/extension';db=base/'documents.sqlite'
    original=json.loads((data/'plan.json').read_text())
    if file_sha256(db)!=original['accepted_database_sha256']:raise AssertionError('Accepted source database changed')
    if file_sha256(base/'code.u16')!=original['raw_token_sha256']['code']:raise AssertionError('Accepted code bytes changed')
    if file_sha256(data/'PRIVATE-training-documents.jsonl')!=json.loads((data/'report.json').read_text())['files']['PRIVATE-training-documents.jsonl']:raise AssertionError('TRAIN membership changed')
    train={json.loads(r)['group'] for r in (data/'PRIVATE-training-documents.jsonl').open()}
    protected={r['group'] for r in json.loads((root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json').read_text())}
    used={r['group'] for r in json.loads((data/'PRIVATE-tune-sources.json').read_text())}
    for f in ('recoverable-dependency-panel-v2/PRIVATE-cases.json','probability-workspace-evaluation-v1/PRIVATE-dependency-metadata.json'):
        used|={r['group'] for r in json.loads((root/f).read_text())}
    excluded=train|protected|used
    tk=Tokenizer.from_file(str(a.tokenizer))
    if file_sha256(a.tokenizer)!=plan['tokenizer_sha256']:raise AssertionError('Tokenizer changed')
    raw=np.memmap(base/'code.u16',mode='r',dtype='<u2');conn=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    inputs=[];targets=[];masks=[];metadata=[];groups=set();counts=collections.Counter();kind_counts=collections.Counter();depth_counts=collections.Counter()
    for identity,group,offset,length in conn.execute("select id,source_group,offset,length from documents where domain='code' and split='train' order by id"):
        if group in excluded or group in groups or int(group[:8],16)%10!=0:continue
        if length<128:continue
        if counts['source_documents_read']>=30000 or len(groups)>=384:break
        counts['source_documents_read']+=1;n=min(length,16384);ids=np.asarray(raw[offset:offset+n]).tolist();counts['source_tokens_read']+=n
        text=tk.decode(ids,skip_special_tokens=False);enc=tk.encode(text,add_special_tokens=False)
        if enc.ids!=ids:counts['token_roundtrip_rejected']+=1;continue
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');ast.parse(text)
            counts['parser_warnings']+=len(ws);pair=paired_spans(text,enc.offsets,identity)
        except (SyntaxError,ValueError,RecursionError,STACK.tokenize.TokenError,IndentationError):counts['parser_rejected_unknown']+=1;continue
        counts['parse_accepted']+=1
        if counts['source_documents_read']%64==0 and (time.time()-started>900 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3):raise RuntimeError('Preparation resource bound exceeded')
        if pair is None:continue
        groups.add(group);counts['paired_source_documents']+=1
        for category,r in zip(('long','short'),pair):
            t=r['target_token'];prefix_start=max(0,t-512);prefix=ids[prefix_start:t]
            if len(prefix)<32 or r['opener_token']<prefix_start:raise AssertionError('Causal opener removed')
            x=np.full(512,2,dtype=np.uint16);y=np.full(512,2,dtype=np.uint16);mask=np.zeros(512,dtype=np.bool_)
            x[:len(prefix)]=prefix;y[:len(prefix)]=ids[prefix_start+1:t+1];mask[len(prefix)-1]=True
            if int(y[len(prefix)-1])!=ids[t]:raise AssertionError('Target must be original source token')
            inputs.append(x);targets.append(y);masks.append(mask)
            metadata.append(dict(id=hashlib.sha256((identity+':'+str(t)+':'+category).encode()).hexdigest(),parent_id=identity,group=group,kind='structural_'+category,domain='code',cohort='prospective-source-separated-development',source_token_sha256=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest(),source_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),source_prefix_start=prefix_start,target_token=t,prefix_tokens=len(prefix),target_tokens=1,opener_token=r['opener_token'],opener_distance=r['distance'],nesting=r['depth'],depth_band=r['depth_band'],closer_type=r['kind'],target_start=len(prefix)-1))
            kind_counts[category+'|'+r['kind']]+=1;depth_counts[category+'|'+str(r['depth_band'])]+=1
        if counts['source_documents_read']%64==0:
            (out/'progress.json').write_text(json.dumps(dict(documents=counts['source_documents_read'],groups=len(groups),seconds=time.time()-started)))
            if time.time()-started>900 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise RuntimeError('Preparation resource bound exceeded')
    conn.close()
    if not inputs:raise RuntimeError('No eligible source pairs;not a model failure')
    np.save(out/'inputs.npy',np.stack(inputs));np.save(out/'targets.npy',np.stack(targets));np.save(out/'mask.npy',np.stack(masks));(out/'PRIVATE-metadata.json').write_text(json.dumps(metadata))
    if groups&excluded or len(metadata)!=2*len(groups):raise AssertionError('Source separation or pairing differs')
    result=dict(status='FROZEN_STRUCTURAL_DEPENDENCY_PANEL' if len(groups)>=128 else 'INSUFFICIENT_STRUCTURAL_SOURCE_YIELD',counts=dict(counts),paired_source_groups=len(groups),cases=len(metadata),kind_counts=dict(kind_counts),depth_counts=dict(depth_counts),source_group_overlap=0,protected_group_overlap=0,previous_evaluation_group_overlap=0,model_scores=0,optimizer_updates=0,assigned_chips=0,
      wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=file_sha256(out/'plan.json'),files={f.name:file_sha256(f) for f in out.iterdir() if f.suffix=='.npy' or f.name=='PRIVATE-metadata.json'},
      scope='New development sources,not terminal confirmation. One authored long/short closing-delimiter pair per source group,matched closer type and nesting band. Targets are unchanged source tokens;original intervening prefix retained. Selection uses syntax only. No source execution or calculated answer. Structural prediction is not a complete reasoning test and local shortcuts remain possible. This preparation admits no fitting or scoring;any readout requires a prospective frozen comparison. It does not replace the ongoing broad-loss primary.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(public/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);main(p.parse_args())
