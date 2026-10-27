"""New-tokenizer views of already admitted natural sources;no model scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import time
from collections import Counter

from transformers import AutoTokenizer
from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.training import file_sha256
from scripts.prepare_natural import rows

PUBLIC=Path('results/workspace_source_views_20261027')


def anchor_positions(length,document_id,destination):
    if length<160:return []
    rng=random.Random(int(hashlib.sha256((document_id+':7403').encode()).hexdigest(),16))
    if destination=='dev':return sorted(rng.sample(range(128,length-31),min(16,length-159)))
    anchors=[first+rng.randrange(min(256,length-31-first)) for first in range(128,length-31,256)]
    cap=32 if destination=='train' else 16
    return sorted(rng.sample(anchors,cap) if len(anchors)>cap else anchors)


def run(root):
    started=time.time();out=root/'workspace-source-views-v1';base=root/'workspace-olmo2-stage1-base-v1'
    if os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('CPU source preparation only')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every source preparation')
    if shutil.disk_usage(root).free<36*1024**3:raise RuntimeError('Whole view allowance plus protected reserve')
    raw=json.loads((root/'natural-append-capacity-v4/raw-source-map.json').read_text())
    receipts={d:next(v for v in raw if v['domain']==d) for d in ['web','math','code']}
    dbs=[]
    for folder in ['natural-append-3250m','natural-append-6500m-v2','natural-append-17000m-v3','natural-append-capacity-v4']:
        p=root/folder/'extension/documents.sqlite'
        if not p.exists():p=root/folder/'documents.sqlite'
        if p.exists():dbs.append(p)
    if len(dbs)!=4:raise AssertionError('Complete four-generation admitted-document metadata required')
    original=json.loads(Path('/run/user/1000/autonomous-compute/data-v1/corpus.json').read_text())
    original_by_id={v['id']:v for v in original}
    excluded=[]
    for p in [root/'workspace-reader-bootstrap-v1/train-views.json',root/'workspace-reader-qualification-v1/views.json']:
        excluded+=json.loads(p.read_text())
    excluded+= [json.loads((root/'workspace-policy-gradient-reference-v1/private-source-view.json').read_text())]
    excluded_groups={v['group'] for v in excluded}
    quotas={'train':{'web':39360,'math':13120,'code':13120},'tune':{'web':300,'math':100,'code':100},'dev':{'web':600,'math':200,'code':200}}
    out.mkdir(mode=0o700);PUBLIC.mkdir()
    plan=dict(registered_unix=time.time(),purpose='SOURCE_VIEWS_ONLY_NO_SCIENTIFIC_FIT_OR_MODEL_SELECTION',
        source_sha256={n:file_sha256(n) for n in ['scripts/prepare_workspace_source_views.py','scripts/prepare_natural.py','compute_trainium/data.py']},
        raw_receipts={d:v['receipt'] for d,v in receipts.items()},accepted_database_sha256={p.parent.parent.name if p.parent.name=='extension' else p.parent.name:file_sha256(p) for p in dbs},
        original_acceptance_manifest_sha256=file_sha256(Path('/run/user/1000/autonomous-compute/data-v1/corpus.json')),
        base_report_sha256=file_sha256(base/'report.json'),quota_by_split_domain=quotas,prefix_tokens=128,ordered_future_tokens=32,
        training_anchor_spacing=256,maximum_document_tokens=16384,maximum_source_characters=2000000,
        maximum_train_views_per_document=32,maximum_tune_views_per_document=16,maximum_dev_views_per_document=16,
        selection_seed=7403,tune_group_rule='SHA256(source_group) first8hex mod10 ==0 among accepted TRAIN groups;all other accepted TRAIN groups train.Exclude every prior calibration group.',
        dev_rule='Previously consumed original DEV documents only;up to16 distinct uniform eligible positions per document.Source-group clustered inference required,not independent windows.No TEST processing or confirmation.',
        scope='Authentic source substrings re-tokenized with pinned OLMo tokenizer. No generated targets,teacher,reasoning labels or loss-based selection. Full four-generation accepted TRAIN ID membership/source-group/domain checked;old DEV accepted IDs frozen. Quotas are requested caps;if yields fall short,stop without replacing by unregistered data. This is a new view protocol,not unchanged raw-text packing or the old exact source-token prefix.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_SOURCE_VIEWS',optimizer_updates=0,assigned_chips=0)
    streams={};connections=[];counts={s:Counter() for s in quotas};groups={s:set() for s in quotas};seen=set();rejected=Counter();at_cap=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<514*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected source-preparation reserve')
        if time.time()-started>1800:raise TimeoutError('Bounded source-view extraction')
    try:
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
        for p in dbs:
            c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);c.execute('pragma cache_size=-8000');connections.append(c)
        streams={s:(out/(s+'-views.jsonl')).open('x') for s in quotas}
        for domain,receipt in receipts.items():
            raw_file=Path(receipt['path'])
            if file_sha256(raw_file)!=receipt['receipt']['sha256']:raise AssertionError('Pinned original source cache changed')
            for row in rows(raw_file):
                guard()
                if all(counts[s][domain]>=quotas[s][domain] for s in quotas):break
                text=row.get('text',row.get('content',''));key=content_hash(text)
                if key in seen:continue
                source=row.get('repo_name','') if domain=='code' else row.get('url','')
                try:group=source_group(domain,source)
                except ValueError:rejected['invalid_source_identity']+=1;continue
                group_hash=hashlib.sha256(group.encode()).hexdigest();split=split_for_group(group)
                if split=='test':continue
                accepted=None
                if key in original_by_id:
                    old=original_by_id[key]
                    if old['domain']!=domain or old['group']!=group:
                        rejected['alternate_source_for_original_content']+=1;continue
                    accepted=old['split']
                elif split=='train':
                    for c in connections:
                        accepted_row=c.execute('select domain,source_group,split from documents where id=?',(key,)).fetchone()
                        if accepted_row:
                            if tuple(accepted_row)!=(domain,group,'train'):
                                rejected['alternate_source_for_same_content']+=1;break
                            accepted='train';break
                if accepted is None:continue
                if accepted!=split:raise AssertionError('Original split changed')
                if split=='dev':destination='dev'
                else:
                    if group_hash in excluded_groups:rejected['prior_calibration_group']+=1;continue
                    destination='tune' if int(group_hash[:8],16)%10==0 else 'train'
                if counts[destination][domain]>=quotas[destination][domain]:continue
                if len(text)>2000000:rejected['source_character_cap']+=1;continue
                tokens=tokenizer.encode(text,add_special_tokens=False,truncation=True,max_length=16384)
                at_cap+=int(len(tokens)==16384)
                if len(tokens)<160:rejected['short_document']+=1;continue
                anchors=anchor_positions(len(tokens),key,destination)
                for anchor in sorted(anchors):
                    if counts[destination][domain]>=quotas[destination][domain]:break
                    view=dict(domain=domain,id=key,group=group_hash,anchor=anchor,prefix=tokens[anchor-128:anchor],continuation=tokens[anchor:anchor+32])
                    streams[destination].write(json.dumps(view,separators=(',',':'))+'\n');counts[destination][domain]+=1;groups[destination].add(group_hash)
                seen.add(key)
                if sum(sum(v.values()) for v in counts.values())%1000<32:
                    (out/'progress.json').write_text(json.dumps(dict(counts={s:dict(v) for s,v in counts.items()},wall_seconds=time.time()-started)))
        for f in streams.values():f.close()
        streams={}
        if any(groups[a]&groups[b] for a,b in [('train','tune'),('train','dev'),('tune','dev')]):raise AssertionError('Source-group leakage')
        if any(counts[s][d]!=n for s,q in quotas.items() for d,n in q.items()):raise ValueError('Registered view yield shortfall;preserve partial data and do not silently refill')
        result.update(status='PASS_WORKSPACE_NATURAL_SOURCE_VIEWS',source_group_isolation=True,
            files={s:dict(sha256=file_sha256(out/(s+'-views.jsonl')),bytes=(out/(s+'-views.jsonl')).stat().st_size) for s in quotas},
            warning='New-tokenizer source views only.No model scored or trained. DEV was previously consumed and is not fresh confirmation;upstream base contamination cannot be ruled out.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for f in streams.values():f.close()
        for c in connections:c.close()
        result.update(wall_seconds=time.time()-started,counts={s:dict(v) for s,v in counts.items()},source_groups={s:len(v) for s,v in groups.items()},
            documents_at_token_cap=at_cap,rejected=dict(rejected),plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
