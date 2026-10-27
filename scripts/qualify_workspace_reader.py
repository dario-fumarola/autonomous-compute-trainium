"""Zero-update pretrained reader diagnostic on frozen natural TRAIN anchors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.training import file_sha256
from compute_trainium.workspace_learning import (Reader,ForwardLedger,WorkspaceFormat,
    sample_workspaces,score_workspaces,target_log_probs,framed_context)
from scripts.prepare_natural import rows
from scripts.fetch_workspace_stage_one_base import REVISION

PUBLIC=Path('results/workspace_reader_qualification_20261027')


def prepare_views(root,tokenizer):
    raw_map=json.loads((root/'natural-append-capacity-v4/raw-source-map.json').read_text())
    old=Path('/run/user/1000/autonomous-compute/data-v1/corpus.json')
    accepted={v['id'] for v in json.loads(old.read_text()) if v['split']=='train'}
    selected=[];seen=set();receipts=[];generator=torch.Generator().manual_seed(204137)
    for domain,count in [('web',6),('math',2),('code',2)]:
        raw=next(r for r in raw_map if r['domain']==domain);file=Path(raw['path'])
        if file_sha256(file)!=raw['receipt']['sha256']:raise AssertionError('Recorded raw source differs')
        receipts.append(dict(domain=domain,**raw['receipt']));found=0
        for row in rows(file):
            text=row.get('text',row.get('content',''));key=content_hash(text)
            if key not in accepted:continue
            source=row.get('repo_name','') if domain=='code' else row.get('url','')
            group=source_group(domain,source)
            if split_for_group(group)!='train' or group in seen:continue
            tokens=tokenizer.encode(text,add_special_tokens=False)
            if len(tokens)<132:continue
            anchor=int(torch.randint(64,129,(1,),generator=generator))
            selected.append(dict(domain=domain,id=key,group=hashlib.sha256(group.encode()).hexdigest(),
                prefix=tokens[max(0,anchor-128):anchor],targets=tokens[anchor:anchor+4],source_anchor=anchor))
            seen.add(group);found+=1
            if found==count:break
        if found!=count:raise ValueError('Insufficient distinct accepted TRAIN groups')
    return selected,receipts,file_sha256(old)


def run(root):
    started=time.time()
    if os.environ.get('USE_TORCH_XLA')!='0':raise ValueError('Explicit CPU-only environment required')
    out=root/'workspace-reader-qualification-v1';base=root/'workspace-olmo2-stage1-base-v1'
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every qualification')
    base_report=json.loads((base/'report.json').read_text())
    if base_report['status']!='COMPLETE_PINNED_STAGE_ONE_BASE_ACQUISITION' or base_report['revision']!=REVISION:
        raise ValueError('Complete pinned stage-one base required')
    available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
    if available<544*1024**3 or shutil.disk_usage(root).free<37*1024**3:raise RuntimeError('Protected CPU qualification reserves')
    for row in base_report['records']:
        if file_sha256(base/row['file'])!=row['sha256']:raise AssertionError('Pinned base asset changed')
    out.mkdir(mode=0o700);PUBLIC.mkdir();torch.set_num_threads(16)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    views,receipts,source_sha=prepare_views(root,tokenizer)
    (out/'views.json').write_text(json.dumps(views))
    fmt=WorkspaceFormat(tuple(tokenizer.encode('\n\nInternal workspace:\n',add_special_tokens=False)),
                        tuple(tokenizer.encode('\n\nContinue the original text:\n',add_special_tokens=False)))
    plan=dict(registered_unix=time.time(),base_revision=REVISION,base_report_sha256=file_sha256(base/'report.json'),
        source_corpus_sha256=source_sha,raw_sources=receipts,views_sha256=file_sha256(out/'views.json'),
        source_sha256={f:file_sha256(f) for f in ['scripts/qualify_workspace_reader.py','compute_trainium/workspace_learning.py']},
        anchors=10,domain_anchors=dict(web=6,math=2,code=2),group_size=4,workspace_length=8,reward_horizon=4,
        seed=204137,temperature=1.,cpu_threads=16,opening_ids=fmt.opening,closing_ids=fmt.closing,
        maximum_seconds=1800,optimizer_updates=0,assigned_chips=0,test_accessed=False,
        scope='TRAIN-only software/reader qualification,not a learned-workspace or held-out performance result. True-future copy is a labeled privileged positive control only.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_READER_QUALIFICATION',optimizer_updates=0,assigned_chips=0)
    ledger=ForwardLedger();measurements=[]
    def progress(stage,**extra):
        if time.time()-started>1800:raise TimeoutError('Bounded CPU qualification ended')
        (out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,**extra)))
    try:
        progress('loading_pinned_model')
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
                dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
        versions=tuple(p._version for p in model.parameters());parameters=sum(p.numel() for p in model.parameters())
        if parameters!=1484916736:raise AssertionError('Actual base parameter inventory differs')
        reader=Reader(lambda x:model(input_ids=x,use_cache=False).logits,ledger)
        sampled=[]
        for index,view in enumerate(views):
            progress('prefix_only_sampling',anchor=index)
            prefix=torch.tensor([view['prefix']],dtype=torch.long)
            z=sample_workspaces(reader,prefix,format=fmt,length=8,group_size=4,
                generator=torch.Generator().manual_seed(204137+index))
            sampled.append(z)
        torch.save(sampled,out/'private-sampled-workspaces.pt')
        for index,(view,z) in enumerate(zip(views,sampled)):
            progress('reader_scoring',anchor=index)
            c=torch.tensor([view['prefix']],dtype=torch.long);y=torch.tensor([view['targets']],dtype=torch.long)
            own=score_workspaces(reader,c,y,z,format=fmt)
            donor=sampled[(index+1)%len(sampled)][:1]
            if views[(index+1)%len(views)]['group']==view['group']:raise AssertionError('Donor must be an independent source group')
            pause=torch.full((1,8),tokenizer.encode('.',add_special_tokens=False)[0],dtype=torch.long)
            donor_score=score_workspaces(reader,c,y,donor,format=fmt)
            pause_score=score_workspaces(reader,c,y,pause,format=fmt)
            # Privileged TRAIN positive control: never enters sampling,learning or a performance claim.
            privileged=score_workspaces(reader,c,y,y.repeat(1,2),format=fmt)
            row=dict(domain=view['domain'],raw_nll=-float(own['raw_log_probability'].mean()),
                framed_empty_nll=-float(own['framed_log_probability'].mean()),
                own_nll=-float(own['workspace_log_probability'].mean()),
                sampled_gain_mean=float(own['absolute_gain'].mean()),sampled_gain_std=float(own['absolute_gain'].std()),
                donor_nll=-float(donor_score['workspace_log_probability'].mean()),
                pause_nll=-float(pause_score['workspace_log_probability'].mean()),
                privileged_future_copy_gain=float(privileged['absolute_gain'].mean()))
            if index==0:
                context=framed_context(c,z[:1],fmt);inputs=torch.cat([context,y[:,:-1]],dim=1)
                embeds=model.get_input_embeddings()(inputs).detach().requires_grad_(True)
                logits=model(inputs_embeds=embeds,use_cache=False).logits[:,context.shape[1]-1:,:]
                logp=logits.float().log_softmax(-1).gather(-1,y.unsqueeze(-1)).mean()
                grad=torch.autograd.grad(logp,embeds)[0]
                start=len(view['prefix'])+len(fmt.opening)
                row['workspace_embedding_gradient_l2']=float(grad[:,start:start+8].norm())
                result['separate_embedding_gradient_diagnostic']=dict(forward_calls=1,backward_calls=1,
                    batch=1,input_positions=inputs.shape[1],projected_positions=inputs.shape[1],
                    warning='Additional uncached forward/backward;not included in inference-only forward ledger.')
            measurements.append(row)
        if tuple(p._version for p in model.parameters())!=versions:raise AssertionError('Qualification changed a model parameter')
        if not all(torch.isfinite(torch.tensor(v)) for row in measurements for k,v in row.items() if k!='domain'):
            raise ValueError('Nonfinite reader diagnostic')
        result.update(status='COMPLETE_PRETRAINED_READER_QUALIFICATION',parameters=parameters,
            model_parameter_versions_unchanged=True,measurements=measurements,
            content_sensitive_embedding_gradient=measurements[0]['workspace_embedding_gradient_l2'],
            warning='Sensitivity and target-copy control are not learned reasoning or held-out generalization. No actor/scorer optimization or Native qualification yet.')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,forward_ledger=ledger.summary(),plan_sha256=file_sha256(out/'plan.json'))
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
