"""Fixed mature weights:standard FP32 versus stable FP32/FP64 loss gradients."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import torch
from transformers import AutoModelForCausalLM
from compute_trainium.training import file_sha256
from compute_trainium.workspace_cpu_reconstruction import load_streamed_model
from compute_trainium.stable_target_loss import stable_target_log_probs
from compute_trainium.workspace_learning import identity_reader_context,WorkspaceFormat,group_advantages
from compute_trainium.scale_sharded_fixture import relative
PUBLIC=Path('results/workspace_loss_stability_20261027')


def run(root):
    started=time.time();out=root/'workspace-loss-stability-diagnosis-v1';parent=root/'runs/workspace-native-online-throughput-v1-online'
    if os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Explicit CPU diagnosis')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain every diagnosis')
    if shutil.disk_usage(root).free<49*1024**3:raise RuntimeError('Both full gradients plus reserve required')
    out.mkdir();PUBLIC.mkdir();torch.set_num_threads(12)
    inputs=torch.load(parent/'private-mature-inputs.pt',weights_only=True);c,y,z,rewards=[inputs[k] for k in ['prefix','targets','workspace','rewards']]
    if bool((group_advantages(rewards)!=0).any()):raise AssertionError('This bounded diagnosis isolates the recorded zero-policy endpoint only')
    plan=dict(registered_unix=time.time(),native_report_sha256=file_sha256(parent/'report.json'),
        inputs_sha256=file_sha256(parent/'private-mature-inputs.pt'),source_sha256={n:file_sha256(n) for n in ['scripts/diagnose_workspace_loss_stability.py','compute_trainium/stable_target_loss.py','compute_trainium/workspace_cpu_reconstruction.py']},
        optimizer_updates=0,maximum_seconds=900,scope='Same fixed u102 actor and generated notes as failed audit. Policy advantages are exactly zero. Recompute only ordinary/reader gradient with stable FP32 versus stable FP64 target loss. This diagnoses arithmetic and does not repair or certify historical training.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_LOSS_STABILITY_DIAGNOSIS',optimizer_updates=0,assigned_chips=0)
    def progress(stage):
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected diagnosis reserve')
        if time.time()-started>900:raise TimeoutError('Bounded fixed-weight diagnosis')
        (out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    try:
        progress('independent_actor_restore')
        model=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
        load_streamed_model([parent/f'rank{k}-u102.zip' for k in range(64)],model,label='model')
        losses={}
        for mode in ['fp32','fp64']:
            progress(mode+'_loss_gradient');model.zero_grad(set_to_none=True)
            def lp(context,targets):
                targets=targets.expand(context.shape[0],-1);x=torch.cat([context,targets[:,:-1]],1)
                logits=model(x,use_cache=False).logits[:,context.shape[1]-1:,:]
                return stable_target_log_probs(logits.double() if mode=='fp64' else logits,targets)
            raw=-lp(c[:,:1],torch.cat([c[:,1:],y],1)).mean()
            read=-lp(identity_reader_context(c,z,WorkspaceFormat()),y).mean();loss=.5*(raw+read);loss.backward()
            losses[mode]=dict(ordinary=float(raw.detach()),reader=float(read.detach()))
            torch.save({n:p.grad for n,p in model.named_parameters()},out/(mode+'-gradients.pt'))
        progress('complete_gradient_comparison')
        fp64=torch.load(out/'fp64-gradients.pt',weights_only=True,mmap=True);fp32=torch.load(out/'fp32-gradients.pt',weights_only=True,mmap=True)
        standard=torch.load(root/'workspace-online-mature-cpu-audit-v1/failed-independent-gradients.pt',weights_only=True,mmap=True)
        result.update(status='COMPLETE_FIXED_WEIGHT_LOSS_STABILITY_DIAGNOSIS',losses=losses,
            stable_fp32_vs_fp64=relative(fp64,fp32),standard_fp32_vs_stable_fp64=relative(fp64,standard),
            gradient_sha256={m:file_sha256(out/(m+'-gradients.pt')) for m in ['fp32','fp64']},
            warning='No weight update.Same failed-history checkpoint;better loss arithmetic does not retroactively certify it.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
