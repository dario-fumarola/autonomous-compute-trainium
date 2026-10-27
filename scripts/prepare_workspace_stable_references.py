"""Fixed C0 and saturated-weight FP64-loss references;no model updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import torch
from transformers import AutoModelForCausalLM
from compute_trainium.training import file_sha256
from compute_trainium.stable_target_loss import stable_target_log_probs
from compute_trainium.workspace_learning import WorkspaceFormat,identity_reader_context,group_advantages
PUBLIC=Path('results/workspace_stable_references_20261027')


def run(root):
    started=time.time();out=root/'workspace-stable-references-v1'
    if os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('CPU references only')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain every precision reference')
    if shutil.disk_usage(root).free<42*1024**3:raise RuntimeError('Complete reference gradient allowance plus reserve')
    torch.set_num_threads(12);out.mkdir();PUBLIC.mkdir()
    inputs=root/'workspace-identity-gradient-v1/private-objective-inputs.pt'
    state=root/'workspace-reader-bootstrap-v1/checkpoint.pt'
    plan=dict(registered_unix=time.time(),inputs_sha256=file_sha256(inputs),checkpoint_sha256=file_sha256(state),
        source_sha256={n:file_sha256(n) for n in ['scripts/prepare_workspace_stable_references.py','compute_trainium/stable_target_loss.py','compute_trainium/workspace_learning.py']},
        optimizer_updates=0,maximum_seconds=900,scope='HF body remains FP32;only target loss arithmetic uses FP64 as independent oracle. C0 frozen rewards/samples unchanged.Mature reference uses retained fixed-weight FP64 diagnosis with exactly zero policy advantages.No historical optimizer trajectory repaired.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_STABLE_WORKSPACE_REFERENCES',optimizer_updates=0)
    try:
        model=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
        saved=torch.load(state,weights_only=True,mmap=True);model.load_state_dict(saved['model']);del saved
        data=torch.load(inputs,weights_only=True);c,y,z,r=[data[k] for k in ['prefix','targets','workspace','rewards']]
        q=json.loads((root/'workspace-reader-qualification-v1/plan.json').read_text());fmt=WorkspaceFormat(tuple(q['opening_ids']),tuple(q['closing_ids']))
        def lp(context,targets):
            targets=targets.expand(context.shape[0],-1);x=torch.cat([context,targets[:,:-1]],1)
            logits=model(x,use_cache=False).logits[:,context.shape[1]-1:,:].double()
            return stable_target_log_probs(logits,targets)
        raw=-lp(c[:,:1],torch.cat([c[:,1:],y],1)).mean()
        read=-lp(identity_reader_context(c,z,fmt),y).mean()
        opening=c.new_tensor(fmt.opening).reshape(1,-1);policy_ctx=torch.cat([c,opening],1).expand(z.shape[0],-1)
        policy=-(group_advantages(r).double()*lp(policy_ctx,z).sum(-1)).mean()
        (.5*raw+.5*read+.1*policy).backward()
        torch.save({n:p.grad for n,p in model.named_parameters()},out/'c0-gradients.pt')
        mature=root/'workspace-loss-stability-diagnosis-v1/fp64-gradients.pt'
        if json.loads((mature.parent/'report.json').read_text())['status']!='COMPLETE_FIXED_WEIGHT_LOSS_STABILITY_DIAGNOSIS':raise AssertionError('Complete fixed-weight FP64 diagnosis required')
        result.update(status='PASS_WORKSPACE_STABLE_CPU_REFERENCES',losses=dict(ordinary=float(raw.detach()),reader=float(read.detach()),policy=float(policy.detach())),
            c0_gradient_relative='workspace-stable-references-v1/c0-gradients.pt',c0_gradient_sha256=file_sha256(out/'c0-gradients.pt'),
            mature_gradient_relative='workspace-loss-stability-diagnosis-v1/fp64-gradients.pt',mature_gradient_sha256=file_sha256(mature))
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
