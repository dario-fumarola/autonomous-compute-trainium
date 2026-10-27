"""Exact source-token arrays and frozen balanced order from accepted JSON views."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
from compute_trainium.training import file_sha256
from compute_trainium.workspace_view_batches import WorkspaceViewBatches
PUBLIC=Path('results/workspace_view_batches_20261027')


def run(root):
    started=time.time();source=root/'workspace-source-views-v2';out=root/'workspace-view-batches-v1'
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every compiled view')
    receipt=json.loads((source/'report.json').read_text())
    if receipt['status']!='PASS_WORKSPACE_NATURAL_SOURCE_VIEWS' or not receipt['source_group_isolation']:raise AssertionError('Complete isolated source views required')
    out.mkdir(mode=0o700);PUBLIC.mkdir();plan=dict(registered_unix=time.time(),source_report_sha256=file_sha256(source/'report.json'),
        source_sha256={n:file_sha256(n) for n in ['scripts/compile_workspace_view_batches.py','compute_trainium/workspace_view_batches.py']},
        shuffle_seed=7459,pattern=['web','math','web','code','web'],optimizer_updates=0,
        scope='No model scoring or training.Token-for-token compilation from frozen new-tokenizer source views,uint32 arrays for shared mmap;frozen within-domain shuffle then60/20/20 round robin.No epoch wrap.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_VIEW_COMPILATION',optimizer_updates=0)
    records={}
    try:
        for split in ['train','tune','dev']:
            source_file=source/(split+'-views.jsonl')
            if file_sha256(source_file)!=receipt['files'][split]['sha256']:raise AssertionError('Frozen source views changed')
            count=sum(receipt['counts'][split].values());tokens=np.lib.format.open_memmap(out/(split+'-tokens.npy'),mode='w+',dtype=np.uint32,shape=(count,160))
            domains=np.lib.format.open_memmap(out/(split+'-domains.npy'),mode='w+',dtype=np.uint8,shape=(count,));by={d:[] for d in ['web','math','code']}
            with source_file.open() as stream:
                for index,line in enumerate(stream):
                    view=json.loads(line);values=view['prefix']+view['continuation']
                    if len(view['prefix'])!=128 or len(view['continuation'])!=32 or not all(type(x) is int and 0<=x<100352 for x in values):raise ValueError('Authentic token view bounds changed')
                    tokens[index]=values;domains[index]=['web','math','code'].index(view['domain']);by[view['domain']].append(index)
            if index+1!=count:raise AssertionError('Exact view coverage changed')
            rng=random.Random(7459+['train','tune','dev'].index(split))
            for values in by.values():rng.shuffle(values)
            if len(by['web'])!=3*len(by['math']) or len(by['code'])!=len(by['math']):raise ValueError('Exact60/20/20 mixture required')
            order=np.asarray([by[d].pop() for _ in range(len(by['math'])) for d in ['web','math','web','code','web']],dtype=np.uint32)
            np.save(out/(split+'-order.npy'),order);tokens.flush();domains.flush();del tokens,domains
            # Full token-for-token readback and a complete unique permutation.
            mapped=WorkspaceViewBatches(out,split)
            if not np.array_equal(np.sort(mapped.order),np.arange(count)):raise AssertionError('Source order duplicates or drops views')
            with source_file.open() as stream:
                for i,line in enumerate(stream):
                    view=json.loads(line)
                    if not np.array_equal(mapped.tokens[i],np.asarray(view['prefix']+view['continuation'],dtype=np.uint32)):raise AssertionError('Compiled source token differs')
            records[split]=dict(views=count,files={p.name:file_sha256(p) for p in out.glob(split+'-*.npy')})
        result.update(status='PASS_WORKSPACE_SHARED_VIEW_BATCHES',records=records,exact_token_readback=True,
            warning='Source-only preparation.No model outcome has selected order or examples.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
