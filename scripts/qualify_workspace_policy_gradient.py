"""Retain a real-base CPU joint gradient;zero actor or scorer updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from compute_trainium.training import file_sha256
from compute_trainium.workspace_learning import (Reader,ForwardLedger,WorkspaceFormat,make_frozen_scorer,
    sample_workspaces,score_workspaces,policy_loss,reader_loss,target_log_probs,group_advantages)
from scripts.bootstrap_workspace_reader import select_views

PUBLIC=Path('results/workspace_policy_gradient_20261027')
DOCUMENT='docs/WORKSPACE_POLICY_GRADIENT_REFERENCE_20261027.md'


def gradient_summary(model):
    sums={};elements={}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite CPU objective gradient')
        family='output_head' if name.startswith('lm_head.') else 'input_embedding' if 'embed_tokens' in name else 'transformer_body'
        norm=float(p.grad.float().norm());sums[family]=sums.get(family,0.)+norm*norm
        elements[family]=elements.get(family,0)+p.numel()
    return {k:dict(l2=v**.5,elements=elements[k]) for k,v in sums.items()}


def run(root):
    started=time.time();out=root/'workspace-policy-gradient-reference-v1';base=root/'workspace-olmo2-stage1-base-v1'
    bootstrap=root/'workspace-reader-bootstrap-v1';qualification=root/'workspace-reader-qualification-v1'
    if os.environ.get('USE_TORCH_XLA')!='0':raise ValueError('Explicit CPU qualification required')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every gradient reference')
    if shutil.disk_usage(root).free<42*1024**3:raise RuntimeError('Full6GiB gradient,1GiB margin and35GiB reserve required')
    def guard():
        available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
        if available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected qualification reserve')
        if time.time()-started>900:raise TimeoutError('Bounded zero-update gradient reference ended')
    guard();report=json.loads((bootstrap/'report.json').read_text())
    if report['status']!='COMPLETE_BOUNDED_READER_BOOTSTRAP' or not report['complete_model_optimizer_rng_restore_exact']:
        raise ValueError('Complete calibrated and restored model required')
    if file_sha256(bootstrap/'checkpoint.pt')!=report['checkpoint_sha256']:raise AssertionError('Calibrated state changed')
    torch.set_num_threads(16);torch.manual_seed(415031)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    excluded=json.loads((bootstrap/'train-views.json').read_text())+json.loads((qualification/'views.json').read_text())
    view=select_views(root,tokenizer,excluded)[0];out.mkdir(mode=0o700);PUBLIC.mkdir()
    (out/'private-source-view.json').write_text(json.dumps(view))
    source=json.loads((qualification/'plan.json').read_text());fmt=WorkspaceFormat(tuple(source['opening_ids']),tuple(source['closing_ids']))
    plan=dict(registered_unix=time.time(),bootstrap_report_sha256=file_sha256(bootstrap/'report.json'),
        calibrated_checkpoint_sha256=report['checkpoint_sha256'],private_source_sha256=file_sha256(out/'private-source-view.json'),
        source_sha256={f:file_sha256(f) for f in ['scripts/qualify_workspace_policy_gradient.py','compute_trainium/workspace_learning.py',DOCUMENT]},
        group_size=4,workspace_length=8,horizon=4,temperature=1.,cost_per_token=.001,seed=415031,
        ordinary_weight=.5,reader_weight=.5,policy_weight=.1,optimizer_updates=0,assigned_chips=0,
        maximum_seconds=900,scope=Path(DOCUMENT).read_text())
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_POLICY_GRADIENT_REFERENCE',optimizer_updates=0,assigned_chips=0)
    ledger=ForwardLedger()
    def progress(stage):guard();(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    try:
        progress('loading_calibrated_actor')
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').eval()
        state=torch.load(bootstrap/'checkpoint.pt',weights_only=True,mmap=True);model.load_state_dict(state['model']);del state
        scorer=make_frozen_scorer(model)
        actor_versions=tuple(p._version for p in model.parameters());scorer_versions=tuple(p._version for p in scorer.parameters())
        actor=Reader(lambda x:model(input_ids=x,use_cache=False).logits,ledger)
        frozen=Reader(lambda x:scorer(input_ids=x,use_cache=False).logits,ledger)
        c=torch.tensor([view['prefix']]);y=torch.tensor([view['targets']]);generator=torch.Generator().manual_seed(415031)
        progress('prefix_only_sampling')
        z=sample_workspaces(actor,c,format=fmt,length=8,group_size=4,generator=generator)
        scores=score_workspaces(frozen,c,y,z,format=fmt,cost_per_token=.001)
        torch.save(dict(prefix=c,targets=y,workspace=z,rewards=scores['rewards'],sampling_rng=generator.get_state()),out/'private-objective-inputs.pt')
        progress('policy_only_gradient')
        pg=policy_loss(actor,c,z,scores['rewards'],format=fmt);pg.backward()
        result['policy_only_gradient']=gradient_summary(model);model.zero_grad(set_to_none=True)
        progress('joint_objective_gradient')
        raw=-target_log_probs(actor,c[:,:1],torch.cat([c[:,1:],y],dim=1),phase='joint_ordinary_ntp').mean()
        read=reader_loss(actor,c,y,z,format=fmt);pg=policy_loss(actor,c,z,scores['rewards'],format=fmt)
        total=.5*raw+.5*read+.1*pg
        if not torch.isfinite(total):raise ValueError('Nonfinite joint objective')
        total.backward();result['joint_gradient']=gradient_summary(model)
        if (any(p.grad is not None for p in scorer.parameters()) or tuple(p._version for p in model.parameters())!=actor_versions
                or tuple(p._version for p in scorer.parameters())!=scorer_versions):raise AssertionError('Frozen scorer or zero-update actor changed')
        progress('saving_full_cpu_gradient')
        temporary=out/'gradients.pt.incomplete';torch.save({name:p.grad for name,p in model.named_parameters()},temporary)
        temporary.rename(out/'gradients.pt')
        result.update(status='COMPLETE_WORKSPACE_POLICY_GRADIENT_REFERENCE',scorer_gradients_absent=True,
            actor_and_scorer_parameter_versions_unchanged=True,losses=dict(ordinary=float(raw.detach()),reader=float(read.detach()),
            policy=float(pg.detach()),joint=float(total.detach())),absolute_gains=scores['absolute_gain'].tolist(),
            rewards=scores['rewards'].tolist(),advantages=group_advantages(scores['rewards']).tolist(),
            full_gradient_sha256=file_sha256(out/'gradients.pt'),full_gradient_bytes=(out/'gradients.pt').stat().st_size,
            backward_calls=2,warning='Real-base gradient qualification only;no writer policy update,held-out benefit or efficiency claim.')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,forward_ledger=ledger.summary(),plan_sha256=file_sha256(out/'plan.json'))
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
