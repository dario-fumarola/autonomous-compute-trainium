"""Freeze random natural windows and retokenized original dependency cases.

No model is read or scored. Original first-window tuning arrays remain intact.
"""
import argparse,collections,hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np


def main(root):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    from compute_trainium.natural_code_context import token_span
    from compute_trainium.training import file_sha256
    out=root/'probability-workspace-evaluation-v1';out.mkdir(exist_ok=False);start=time.time()
    data=root/'probability-workspace-data-v1';source=root/'natural-append-3250m/extension'
    meta=json.loads((data/'PRIVATE-tune-sources.json').read_text())
    train_groups={json.loads(line)['group'] for line in (data/'PRIVATE-training-documents.jsonl').open()}
    protected={r['group'] for r in json.loads((root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json').read_text())}
    plan=dict(purpose='Evaluation source preparation only; no model scores',registered_unix=start,
        broad_groups_sha256=file_sha256(data/'PRIVATE-tune-sources.json'),old_dependency_sha256=file_sha256(root/'recoverable-dependency-panel-v2/PRIVATE-cases.json'),
        selection_rule='Same384 frozen tuning groups; SHA256(document_id:probability-window-9511) selects one uniform eligible start within first16384 admitted source tokens.',
        dependency_rule='All340 previously verified cases,retokenized from exactly round-tripped source token strings; reject train/reserve overlap,nonaligned target edges or full prefix+answer beyond512. No truncation that could remove the antecedent.',
        original_first_window_tuning_panel_scored=False,context=512,model_scores=0)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    arrays={d:np.memmap(source/(d+'.u16'),dtype='<u2',mode='r') for d in ('web','math','code')}
    conn=sqlite3.connect('file:'+str(source/'documents.sqlite')+'?mode=ro',uri=True)
    tokens=[];records=[]
    for row in meta:
        got=conn.execute('select domain,source_group,offset,length from documents where id=?',(row['id'],)).fetchone()
        if got is None or got[:2]!=(row['domain'],row['group']):raise AssertionError('Original accepted tuning identity changed')
        domain,group,offset,length=got
        if group in train_groups|protected:raise AssertionError('Broad panel source leakage')
        cap=min(length,16384);start_token=int(hashlib.sha256((row['id']+':probability-window-9511').encode()).hexdigest(),16)%(cap-512)
        view=arrays[domain][offset+start_token:offset+start_token+513].copy()
        if len(view)!=513:raise AssertionError('Original source bounds')
        tokens.append(view);records.append(dict(**row,source_window_start=start_token,kind='broad',target_tokens=512))
    np.save(out/'broad-tokens.npy',np.stack(tokens));(out/'PRIVATE-broad-metadata.json').write_text(json.dumps(records));conn.close()
    oldtok=AutoTokenizer.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False)
    tok_path=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json');newtok=Tokenizer.from_file(str(tok_path))
    plan['tokenizer_sha256']=file_sha256(tok_path);rejected=collections.Counter();cases=[];inputs=[];targets=[];masks=[]
    for row in json.loads((root/'recoverable-dependency-panel-v2/PRIVATE-cases.json').read_text()):
        if row['group'] in train_groups|protected:rejected['source_group_overlap']+=1;continue
        ids=row['full_prefix']+row['answer']
        prefix=oldtok.decode(row['full_prefix'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        answer=oldtok.decode(row['answer'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        text=oldtok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if text!=prefix+answer or oldtok.encode(text,add_special_tokens=False)!=ids:
            rejected['old_tokenizer_roundtrip']+=1;continue
        enc=newtok.encode(text,add_special_tokens=False);span=token_span(text,enc.offsets,len(prefix),len(text))
        if span is None or span[0]<1:rejected['new_target_alignment']+=1;continue
        lo,hi=span
        if hi-1>512 or hi-lo>16:rejected['context_or_target_cap']+=1;continue
        x=np.full(512,2,dtype=np.uint16);y=np.full(512,2,dtype=np.uint16);m=np.zeros(512,dtype=np.bool_)
        x[:hi-1]=enc.ids[:hi-1];y[:hi-1]=enc.ids[1:hi];m[lo-1:hi-1]=True
        inputs.append(x);targets.append(y);masks.append(m)
        cases.append({k:row[k] for k in ('id','parent_id','group','kind','cohort','source_utf8_sha256')}|dict(target_tokens=hi-lo,target_start=lo-1,source_prefix_tokens=lo))
    np.save(out/'dependency-inputs.npy',np.stack(inputs));np.save(out/'dependency-targets.npy',np.stack(targets));np.save(out/'dependency-mask.npy',np.stack(masks))
    (out/'PRIVATE-dependency-metadata.json').write_text(json.dumps(cases))
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    report=dict(status='FROZEN_PROBABILITY_EVALUATION',broad_groups=len(records),dependency_cases=len(cases),dependency_groups=len({r['group'] for r in cases}),
        dependency_counts=dict(collections.Counter(r['kind'] for r in cases)),rejected=dict(rejected),source_group_overlap=0,protected_group_overlap=0,
        model_scores=0,assigned_chips=0,wall_seconds=time.time()-start,
        files={p.name:file_sha256(p) for p in out.iterdir() if p.suffix=='.npy' or p.name.startswith('PRIVATE')},
        scope='Development panel,not fresh terminal confirmation. Arithmetic targets and code names remain authentic source substrings. Code naming is not by itself multi-step reasoning. Whole-prefix retention avoids deleting the original antecedent.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
