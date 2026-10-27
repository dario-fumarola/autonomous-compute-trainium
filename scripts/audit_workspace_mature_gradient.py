"""Independent full HF gradient at the completed online qualification endpoint."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import time
import torch
from transformers import AutoModelForCausalLM

from compute_trainium.training import file_sha256
from compute_trainium.workspace_cpu_reconstruction import load_streamed_model
from compute_trainium.workspace_learning import (Reader,WorkspaceFormat,identity_reader_context,
    score_workspaces,reader_loss,policy_loss,target_log_probs)

PUBLIC=Path('results/workspace_online_mature_audit_20261027')


def run(root):
    started=time.time();out=root/'workspace-online-mature-cpu-audit-v1';native=root/'runs/workspace-native-online-throughput-v1-online'
    if os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Independent CPU reference only')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every mature audit')
    receipt=json.loads((native/'report.json').read_text())
    if receipt['status']!='PASS_WORKSPACE_ONLINE_THROUGHPUT_EXECUTION_PENDING_MATURE_AUDIT':raise AssertionError('Complete fixed-update Native endpoint required')
    if shutil.disk_usage(root).free<42*1024**3:raise RuntimeError('Diagnostic6GiB gradient allowance plus protected reserve')
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected CPU/file reserve')
        if time.time()-started>1200:raise TimeoutError('Bounded complete mature audit')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<552*1024**3:raise RuntimeError('Complete40GiB CPU working allowance and512GiB reserve')
    out.mkdir();PUBLIC.mkdir();torch.set_num_threads(12)
    source=['scripts/audit_workspace_mature_gradient.py','compute_trainium/workspace_cpu_reconstruction.py','compute_trainium/workspace_learning.py']
    shards=[native/f'rank{rank}-u102.zip' for rank in range(64)]
    gradient_files=[native/f'rank{rank}-mature-gradients.pt' for rank in range(64)]
    plan=dict(registered_unix=time.time(),native_report_sha256=file_sha256(native/'report.json'),
        shard_sha256=[file_sha256(p) for p in shards],gradient_sha256=[file_sha256(p) for p in gradient_files],
        inputs_sha256=file_sha256(native/'private-mature-inputs.pt'),source_sha256={n:file_sha256(n) for n in source},
        torch_version=torch.__version__,maximum_seconds=1200,optimizer_updates=0,
        scope='Independent HF eager full179-parameter gradient at the actual saved u102 actor/scorer. Shared numerical audit workspace/rewards from Native,independent frozen scores checked separately. Source is one repeated TRAIN anchor.2% every named parameter criterion unchanged. No scientific efficiency or transfer claim.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_ONLINE_MATURE_CPU_AUDIT',optimizer_updates=0,assigned_chips=0)
    def progress(stage):guard();(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    try:
        base=root/'workspace-olmo2-stage1-base-v1'
        progress('independent_hf_actor_restore')
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
        load_streamed_model(shards,model,label='model')
        progress('independent_hf_frozen_scorer_restore')
        scorer=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
        load_streamed_model(shards,scorer,label='scorer')
        inputs=torch.load(native/'private-mature-inputs.pt',weights_only=True);c,y,z,rewards=[inputs[k] for k in ['prefix','targets','workspace','rewards']]
        native_plan=json.loads((root/'workspace-native-online-throughput-v1/plan.json').read_text())
        fmt=WorkspaceFormat(tuple(native_plan['opening_ids']),tuple(native_plan['closing_ids']))
        actor=Reader(lambda x:model(x,use_cache=False).logits);frozen=Reader(lambda x:scorer(x,use_cache=False).logits)
        progress('independent_frozen_reader_scores')
        scores=score_workspaces(frozen,c,y,z,format=fmt,cost_per_token=.001,context_builder=identity_reader_context)
        score_error=float((scores['rewards']-rewards).abs().max())
        if score_error>.002:raise AssertionError('Mature independent frozen score mismatch')
        versions=tuple(p._version for p in model.parameters())
        raw=-target_log_probs(actor,c[:,:1],torch.cat([c[:,1:],y],1),phase='ordinary').mean()
        read=reader_loss(actor,c,y,z,format=fmt,context_builder=identity_reader_context);pg=policy_loss(actor,c,z,rewards,format=fmt)
        progress('full_independent_joint_gradient');(.5*raw+.5*read+.1*pg).backward()
        progress('all_named_native_shard_comparison');gradients=[torch.load(p,weights_only=True,mmap=True) for p in gradient_files]
        errors={};diff_total=ref_total=0.
        for name,p in model.named_parameters():
            error=reference=0.
            for rank in range(64):
                expected=p.grad.chunk(64,0)[rank].double();actual=gradients[rank][name].double()
                error+=float((actual-expected).square().sum());reference+=float(expected.square().sum())
            errors[name]=(error/max(reference,1e-30))**.5;diff_total+=error;ref_total+=reference
        if max(errors.values())>.02:
            torch.save({n:p.grad for n,p in model.named_parameters()},out/'failed-independent-gradients.pt')
            raise AssertionError('Complete mature gradient exceeds unchanged2% named limit')
        if any(p.grad is not None for p in scorer.parameters()) or versions!=tuple(p._version for p in model.parameters()):raise AssertionError('Frozen scorer/zero-update actor invariant')
        result.update(status='PASS_WORKSPACE_ONLINE_MATURE_CPU_AUDIT',parameters_checked=len(errors),
            gradient_relative_l2=(diff_total/max(ref_total,1e-30))**.5,largest_named_relative_l2=max(errors.values()),
            worst_named_parameter=max(errors,key=errors.get),frozen_score_max_abs_error=score_error,
            losses=dict(ordinary=float(raw.detach()),reader=float(read.detach()),policy=float(pg.detach())),
            scorer_gradients_absent=True,actor_versions_unchanged=True,
            warning='Mature numerical qualification only. Repeated TRAIN source;no useful-writing,generalization,from-scratch or matched-compute finding.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
