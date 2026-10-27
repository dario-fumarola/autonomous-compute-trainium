"""One fixed thirty-update CPU reader calibration;no policy optimization."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.training import file_sha256
from compute_trainium.workspace_learning import (Reader,ForwardLedger,WorkspaceFormat,
    sample_workspaces,score_workspaces,target_log_probs,reader_loss)
from scripts.prepare_natural import rows

PUBLIC=Path('results/workspace_reader_bootstrap_20261027')
DOCUMENT='docs/WORKSPACE_READER_BOOTSTRAP_20261027.md'


def exact_equal(left,right):
    if isinstance(left,torch.Tensor):
        return (isinstance(right,torch.Tensor) and left.dtype==right.dtype
                and left.shape==right.shape and torch.equal(left,right))
    if type(left)!=type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(exact_equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(exact_equal(a,b) for a,b in zip(left,right))
    return left==right


def learning_rate(update):
    if not 1<=update<=30:raise ValueError('Exactly the declared thirty updates')
    if update<=3:return 1e-5*update/3
    if update<=27:return 1e-5
    return 1e-5*.5*(1+math.cos(math.pi*(update-27)/3))


def select_views(root,tokenizer,excluded):
    corpus=json.loads(Path('/run/user/1000/autonomous-compute/data-v1/corpus.json').read_text())
    accepted={v['id'] for v in corpus if v['split']=='train'}
    raw=json.loads((root/'natural-append-capacity-v4/raw-source-map.json').read_text())
    seen={v['group'] for v in excluded};selected=[]
    for domain,count in [('web',18),('math',6),('code',6)]:
        receipt=next(v for v in raw if v['domain']==domain);file=Path(receipt['path'])
        if file_sha256(file)!=receipt['receipt']['sha256']:raise AssertionError('Original raw source changed')
        found=0
        for row in rows(file):
            text=row.get('text',row.get('content',''));key=content_hash(text)
            if key not in accepted:continue
            source=row.get('repo_name','') if domain=='code' else row.get('url','')
            group=source_group(domain,source)
            if split_for_group(group)!='train':continue
            group_sha=__import__('hashlib').sha256(group.encode()).hexdigest()
            if group_sha in seen:continue
            tokens=tokenizer.encode(text,add_special_tokens=False)
            if len(tokens)<132:continue
            selected.append(dict(domain=domain,id=key,group=group_sha,prefix=tokens[:128],targets=tokens[128:132]))
            seen.add(group_sha);found+=1
            if found==count:break
        if found!=count:raise ValueError('Insufficient disjoint natural TRAIN groups')
    # Fixed round robin18/6/6,prevent a domain-only tail from defining cooldown.
    by={d:[v for v in selected if v['domain']==d] for d in ['web','math','code']}
    return [by[d].pop(0) for _ in range(6) for d in ['web','math','web','code','web']]


def run(root):
    started=time.time();out=root/'workspace-reader-bootstrap-v1';base=root/'workspace-olmo2-stage1-base-v1'
    if os.environ.get('USE_TORCH_XLA')!='0':raise ValueError('CPU qualification environment required')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every calibration attempt')
    if shutil.disk_usage(root).free<56*1024**3:raise RuntimeError('Full20GiB checkpoint plus margin and35GiB reserve required')
    def guard():
        available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
        if available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected CPU/output reserve')
        if time.time()-started>1800:raise TimeoutError('Bounded thirty-update calibration ended')
    guard();torch.set_num_threads(16);torch.manual_seed(315017)
    qualification=root/'workspace-reader-qualification-v1';prior=json.loads((qualification/'report.json').read_text())
    if prior['status']!='COMPLETE_PRETRAINED_READER_QUALIFICATION':raise ValueError('Complete reader qualification required')
    eval_views=json.loads((qualification/'views.json').read_text());source_plan=json.loads((qualification/'plan.json').read_text())
    if (file_sha256(qualification/'plan.json')!=prior['plan_sha256']
            or file_sha256(qualification/'views.json')!=source_plan['views_sha256']
            or file_sha256(base/'report.json')!=source_plan['base_report_sha256']):
        raise AssertionError('Qualified base or source views changed')
    for row in json.loads((base/'report.json').read_text())['records']:
        if file_sha256(base/row['file'])!=row['sha256']:raise AssertionError('Pinned base asset changed')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    train=select_views(root,tokenizer,eval_views);out.mkdir(mode=0o700);PUBLIC.mkdir()
    (out/'train-views.json').write_text(json.dumps(train))
    fmt=WorkspaceFormat(tuple(source_plan['opening_ids']),tuple(source_plan['closing_ids']))
    plan=dict(registered_unix=time.time(),base_revision=source_plan['base_revision'],source_view_sha256=file_sha256(out/'train-views.json'),
        evaluation_view_sha256=file_sha256(qualification/'views.json'),original_qualification_sha256=file_sha256(qualification/'report.json'),
        source_sha256={f:file_sha256(f) for f in ['scripts/bootstrap_workspace_reader.py','compute_trainium/workspace_learning.py',DOCUMENT]},
        updates=30,domain_updates=dict(web=18,math=6,code=6),group_size=4,workspace_length=8,reward_horizon=4,
        ordinary_weight=.5,reader_weight=.5,peak_lr=1e-5,betas=[.9,.95],epsilon=1e-8,weight_decay=.1,gradient_clip=1.,
        seed=315017,maximum_seconds=1800,assigned_chips=0,policy_gradient_updates=0,scope=Path(DOCUMENT).read_text())
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_READER_BOOTSTRAP',assigned_chips=0,policy_gradient_updates=0)
    ledger=ForwardLedger();updates=[]
    def progress(stage,**extra):
        guard();(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,**extra)))
    try:
        progress('loading_model')
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
                dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
        reader=Reader(lambda x:model(input_ids=x,use_cache=False).logits,ledger);generator=torch.Generator().manual_seed(315017)
        samples=[]
        for i,view in enumerate(train):
            progress('initial_prefix_only_sampling',anchor=i)
            samples.append(sample_workspaces(reader,torch.tensor([view['prefix']]),format=fmt,length=8,group_size=4,generator=generator))
        torch.save(samples,out/'private-initial-workspaces.pt');model.requires_grad_(True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
        for update,(view,z) in enumerate(zip(train,samples),1):
            progress('reader_ntp_update',update=update)
            c=torch.tensor([view['prefix']]);y=torch.tensor([view['targets']]);optimizer.zero_grad(set_to_none=True)
            raw_targets=torch.cat([c[:,1:],y],dim=1)
            raw_loss=-target_log_probs(reader,c[:,:1],raw_targets,phase='bootstrap_ordinary_ntp').mean()
            worked_loss=reader_loss(reader,c,y,z,format=fmt);loss=.5*(raw_loss+worked_loss)
            if not torch.isfinite(loss):raise ValueError('Nonfinite calibration loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(update)
            optimizer.step();updates.append(dict(update=update,domain=view['domain'],loss=float(loss.detach()),
                raw_loss=float(raw_loss.detach()),reader_loss=float(worked_loss.detach()),gradient_norm=float(norm),lr=learning_rate(update)))
            (out/'updates.json').write_text(json.dumps(updates))
        optimizer.zero_grad(set_to_none=True);initial=torch.load(qualification/'private-sampled-workspaces.pt',weights_only=True)
        measurements=[]
        for i,(view,z) in enumerate(zip(eval_views,initial)):
            progress('held_out_from_calibration_readout',anchor=i)
            c=torch.tensor([view['prefix']]);y=torch.tensor([view['targets']])
            fixed=score_workspaces(reader,c,y,z,format=fmt)
            fresh=sample_workspaces(reader,c,format=fmt,length=8,group_size=4,
                generator=torch.Generator().manual_seed(204137+i))
            new=score_workspaces(reader,c,y,fresh,format=fmt)
            measurements.append(dict(domain=view['domain'],raw_nll=-float(fixed['raw_log_probability'].mean()),
                framed_empty_nll=-float(fixed['framed_log_probability'].mean()),fixed_workspace_nll=-float(fixed['workspace_log_probability'].mean()),
                fresh_workspace_nll=-float(new['workspace_log_probability'].mean()),fresh_reward_std=float(new['absolute_gain'].std())))
        progress('complete_state_save')
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                   sampling_rng=generator.get_state(),updates=30,plan_sha256=file_sha256(out/'plan.json'))
        temporary=out/'checkpoint.pt.incomplete';torch.save(state,temporary);temporary.rename(out/'checkpoint.pt');del state
        saved=torch.load(out/'checkpoint.pt',weights_only=True)
        with torch.no_grad():
            for p in model.parameters():p.reshape(-1)[0].add_(1.)
            for values in optimizer.state.values():
                for value in values.values():
                    if isinstance(value,torch.Tensor):value.reshape(-1)[0].add_(1.)
        torch.rand(7);generator.manual_seed(1)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);generator.set_state(saved['sampling_rng'])
        if not (exact_equal(model.state_dict(),saved['model']) and exact_equal(optimizer.state_dict(),saved['optimizer'])
                and torch.equal(torch.get_rng_state(),saved['torch_rng']) and torch.equal(generator.get_state(),saved['sampling_rng'])):
            raise AssertionError('Full restored state differs')
        result.update(status='COMPLETE_BOUNDED_READER_BOOTSTRAP',optimizer_updates=30,measurements=measurements,
            complete_model_optimizer_rng_restore_exact=True,checkpoint_sha256=file_sha256(out/'checkpoint.pt'),
            checkpoint_bytes=(out/'checkpoint.pt').stat().st_size,backward_calls=30,
            warning='One tiny reader calibration,not learned workspace content,transfer,efficiency or from-scratch evidence. No policy gradient.')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        if ('model' in locals() and 'optimizer' in locals() and not (out/'checkpoint.pt').exists()
                and not (out/'checkpoint.pt.incomplete').exists() and shutil.disk_usage(root).free>=55*1024**3):
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                sampling_rng=generator.get_state(),completed_updates=len(updates)),out/'failed-checkpoint.pt')
            result['failed_complete_state_preserved']=True
        raise
    finally:
        result.update(completed_updates=len(updates),updates=updates,wall_seconds=time.time()-started,
            forward_ledger=ledger.summary(),plan_sha256=file_sha256(out/'plan.json'))
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
