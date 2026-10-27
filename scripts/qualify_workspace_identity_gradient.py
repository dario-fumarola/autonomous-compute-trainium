"""Full1.485B identity-reader joint gradient and independent Native-model CPU parity."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoConfig,AutoModelForCausalLM
from compute_trainium.training import file_sha256
from compute_trainium.workspace_learning import (Reader,ForwardLedger,WorkspaceFormat,identity_reader_context,
    make_frozen_scorer,score_workspaces,policy_loss,reader_loss,target_log_probs,group_advantages)
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
from scripts.qualify_workspace_policy_gradient import gradient_summary
from compute_trainium.scale_sharded_fixture import relative

IDENTITY='workspace-identity-gradient-v1'
PUBLIC=Path('results/workspace_identity_gradient_20261027')


def run(root):
    started=time.time();out=root/IDENTITY;base=root/'workspace-olmo2-stage1-base-v1'
    bootstrap=root/'workspace-reader-bootstrap-v1';original=root/'workspace-policy-gradient-reference-v1'
    gates=root/'workspace-identity-reader-v2'
    if os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Explicit CPU reference required')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain every reference')
    if shutil.disk_usage(root).free<43*1024**3:raise RuntimeError('Complete gradient plus protected storage reserve required')
    def guard():
        available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
        if available<536*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Full reference and protected reserves required')
        if time.time()-started>900:raise TimeoutError('Bounded full-model reference')
    check=json.loads((gates/'report.json').read_text())
    if not check['both_point_diagnostic_gates_pass']:raise AssertionError('Both reader interface gates required')
    parent=json.loads((bootstrap/'report.json').read_text())
    if file_sha256(bootstrap/'checkpoint.pt')!=parent['checkpoint_sha256']:raise AssertionError('Exact calibrated weights required')
    torch.set_num_threads(12);torch.manual_seed(415031)
    inputs=torch.load(original/'private-objective-inputs.pt',weights_only=True)
    c,y,z=inputs['prefix'],inputs['targets'],inputs['workspace']
    old_plan=json.loads((root/'workspace-reader-qualification-v1/plan.json').read_text())
    fmt=WorkspaceFormat(tuple(old_plan['opening_ids']),tuple(old_plan['closing_ids']))
    out.mkdir(mode=0o700);PUBLIC.mkdir()
    source=['scripts/qualify_workspace_identity_gradient.py','compute_trainium/workspace_learning.py',
            'compute_trainium/workspace_olmo_candidate.py']
    plan=dict(registered_unix=time.time(),source_sha256={n:file_sha256(n) for n in source},
        calibrated_checkpoint_sha256=parent['checkpoint_sha256'],reader_gates_report_sha256=file_sha256(gates/'report.json'),
        original_prefix_only_samples_sha256=file_sha256(original/'private-objective-inputs.pt'),
        reader_layout='workspace_then_unchanged_prefix_no_reader_prose',G=4,workspace_length=8,horizon=4,
        ordinary_weight=.5,reader_weight=.5,policy_weight=.1,cost_per_token=.001,maximum_seconds=900,
        optimizer_updates=0,scope='Same accepted TRAIN anchor and prefix-only samples from prior zero-update reference. Frozen independent scorer. No policy fit. Complete actual-model FP32 gradient and independent Native-model forward/every-gradient parity.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_IDENTITY_GRADIENT',optimizer_updates=0,assigned_chips=0)
    ledger=ForwardLedger()
    def progress(stage):guard();(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    try:
        progress('loading_independent_hf_actor')
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
        saved=torch.load(bootstrap/'checkpoint.pt',weights_only=True,mmap=True);model.load_state_dict(saved['model']);del saved
        scorer=make_frozen_scorer(model);versions=tuple(p._version for p in model.parameters())
        actor=Reader(lambda x:model(x,use_cache=False).logits,ledger);frozen=Reader(lambda x:scorer(x,use_cache=False).logits,ledger)
        scores=score_workspaces(frozen,c,y,z,format=fmt,cost_per_token=.001,context_builder=identity_reader_context)
        torch.save(dict(prefix=c,targets=y,workspace=z,rewards=scores['rewards'],sampling_rng=inputs['sampling_rng']),out/'private-objective-inputs.pt')
        def objective(reader):
            raw=-target_log_probs(reader,c[:,:1],torch.cat([c[:,1:],y],1),phase='ordinary_ntp').mean()
            read=reader_loss(reader,c,y,z,format=fmt,context_builder=identity_reader_context)
            pg=policy_loss(reader,c,z,scores['rewards'],format=fmt)
            return .5*raw+.5*read+.1*pg,dict(ordinary=float(raw.detach()),reader=float(read.detach()),policy=float(pg.detach()))
        progress('independent_hf_joint_gradient');loss,losses=objective(actor);loss.backward()
        result['independent_hf_joint_gradient']=gradient_summary(model)
        if any(p.grad is not None for p in scorer.parameters()) or tuple(p._version for p in model.parameters())!=versions:raise AssertionError('Zero-update actor/frozen scorer invariant')
        torch.save({n:p.grad.detach() for n,p in model.named_parameters()},out/'gradients.pt')
        with torch.no_grad():reference_logits=model(c,use_cache=False).logits[:,-1,:].clone()
        result.update(losses=losses,absolute_gains=scores['absolute_gain'].tolist(),advantages=group_advantages(scores['rewards']).tolist(),
                      scorer_gradients_absent=True,actor_versions_unchanged=True)
        del actor,frozen,model,scorer,loss;gc.collect()
        progress('native_candidate_full_model_gradient')
        config=OlmoWorkspaceConfig.from_hf(AutoConfig.from_pretrained(base,local_files_only=True))
        with torch.device('meta'):candidate=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
        saved=torch.load(bootstrap/'checkpoint.pt',weights_only=True,mmap=True);candidate.load_state_dict(saved['model'],assign=True);del saved
        def native_logits(x):
            cos,sin=rotary_tensors(config,x.shape[1]);return candidate(x,cos,sin)
        with torch.no_grad():actual=native_logits(c)[:,-1,:]
        result['full_model_logits_error']=relative({'logits':reference_logits},{'logits':actual})
        if not torch.allclose(actual,reference_logits,rtol=2e-4,atol=2e-4):raise AssertionError('Actual1.485B model forward mismatch')
        value,native_losses=objective(Reader(native_logits,ledger));value.backward()
        reference=torch.load(out/'gradients.pt',weights_only=True,mmap=True)
        comparison=relative(reference,{n:p.grad for n,p in candidate.named_parameters()})
        if comparison['largest_named_relative_l2']>.0002:raise AssertionError('Complete actual-model CPU gradients disagree')
        result.update(status='PASS_WORKSPACE_IDENTITY_FULL_MODEL_GRADIENT',candidate_losses=native_losses,
            complete_gradient_comparison=comparison,full_gradient_sha256=file_sha256(out/'gradients.pt'),
            full_gradient_bytes=(out/'gradients.pt').stat().st_size,
            objective_inputs_sha256=file_sha256(out/'private-objective-inputs.pt'),backward_calls=2,
            warning='Numerical qualification only. Same TRAIN anchor, zero policy optimization, no useful-thought or efficiency finding.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,forward_ledger=ledger.summary(),plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
