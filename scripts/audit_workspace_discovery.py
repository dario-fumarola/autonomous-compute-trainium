"""Independent eager HF/FP64-loss references for bounded discovery conditions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time


def run(a):
    import torch
    from transformers import AutoModelForCausalLM
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.training import file_sha256
    from compute_trainium.workspace_cpu_reconstruction import load_streamed_model
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    from compute_trainium.workspace_learning import Reader,WorkspaceFormat,target_log_probs,reader_loss,policy_loss,identity_reader_context,score_workspaces
    if os.environ.get('USE_TORCH_XLA')!='0' or managed_running():raise RuntimeError('Independent CPU phase with no active Native owner required')
    initial=a.arm=='initial-ordinary';identity='workspace-discovery-initial-ordinary-v1' if initial else 'workspace-discovery-audit-'+a.arm+'-v1'
    out=a.root/identity;public=Path('results/workspace_discovery_20261027')/identity
    if out.exists() or public.exists():raise FileExistsError('Retain every independent audit')
    started=time.time();out.mkdir();public.mkdir(parents=True);torch.set_num_threads(12)
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<512*1024**3 or shutil.disk_usage(a.root).free<35*1024**3 or time.time()-started>1200:raise RuntimeError('Independent CPU resource/time gate')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<552*1024**3 or shutil.disk_usage(a.root).free<(42 if initial else 36)*1024**3:raise RuntimeError('Full CPU working/artifact allowance required')
    sources=['scripts/audit_workspace_discovery.py','compute_trainium/workspace_cpu_reconstruction.py','compute_trainium/workspace_learning.py','compute_trainium/stable_target_loss.py']
    registration=dict(registered_unix=time.time(),arm=a.arm,optimizer_updates=0,assigned_chips=0,source_sha256={n:file_sha256(n) for n in sources},loss_dtype='float64',body_dtype='float32',named_gradient_limit=.02)
    if initial:
        parent=json.loads((a.root/'workspace-native-online-throughput-v2/plan.json').read_text())
        registration.update(checkpoint_sha256=file_sha256(a.root/parent['checkpoint_relative']),source_batches_report_sha256=file_sha256(a.root/'workspace-view-batches-v1/report.json'))
    for d in [out,public]:(d/'plan.json').write_text(json.dumps(registration,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_DISCOVERY_CPU_AUDIT',arm=a.arm,optimizer_updates=0,assigned_chips=0)
    try:
        base=a.root/'workspace-olmo2-stage1-base-v1'
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
        if initial:
            parent=json.loads((a.root/'workspace-native-online-throughput-v2/plan.json').read_text())
            model.load_state_dict(torch.load(a.root/parent['checkpoint_relative'],weights_only=True,mmap=True)['model'])
            view=WorkspaceViewBatches(a.root/'workspace-view-batches-v1','train').at(0);c,y=view['prefix'],view['continuation']
            actor=Reader(lambda x:model(x,use_cache=False).logits,loss_dtype=torch.float64)
            loss=-target_log_probs(actor,c[:,:1],torch.cat([c[:,1:],y],1),phase='ordinary').mean()
        else:
            plan=json.loads(a.plan.read_text());native=a.root/'runs'/plan.get('arm_runs',{}).get(a.arm,plan['identity']+'-'+a.arm);count=plan['updates'][a.arm]
            report=json.loads((native/'report.json').read_text())
            if report['status']!='COMPLETE_WORKSPACE_DISCOVERY_ARM_PENDING_MATURE_AUDIT':raise AssertionError('Complete frozen endpoint required')
            shards=[native/f'rank{k}-u{count}.zip' for k in range(64)];load_streamed_model(shards,model,label='model')
            inputs=torch.load(native/'private-mature-inputs.pt',weights_only=True);c,y,z,rewards=[inputs[k] for k in ['prefix','continuation','workspace','rewards']]
            fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']))
            actor=Reader(lambda x:model(x,use_cache=False).logits,loss_dtype=torch.float64)
            raw=-target_log_probs(actor,c[:,:1],torch.cat([c[:,1:],y if a.arm=='ordinary' else y[:,:4]],1),phase='ordinary').mean()
            if a.arm=='ordinary':loss=raw
            elif a.arm=='pause':
                context=torch.cat([c,c.new_full((1,8),plan['pause_token_id'])],1)
                loss=.5*raw-.5*target_log_probs(actor,context,y[:,:4],phase='pause_reader').mean()
            else:
                read=reader_loss(actor,c,y[:,:4],z,format=fmt,context_builder=identity_reader_context)
                if a.arm=='workspace':
                    scorer=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
                    load_streamed_model(shards,scorer,label='scorer')
                    frozen=Reader(lambda x:scorer(x,use_cache=False).logits,loss_dtype=torch.float64)
                    scores=score_workspaces(frozen,c,y[:,:4],z,format=fmt,cost_per_token=.001,context_builder=identity_reader_context)
                    error=float((scores['rewards']-rewards).abs().max());result['frozen_reward_max_abs_error']=error
                    if error>.002:raise AssertionError('Independent frozen score mismatch')
                    loss=.5*raw+.5*read+.1*policy_loss(actor,c,z,rewards,format=fmt)
                else:
                    context=torch.cat([c,c.new_tensor(fmt.opening).reshape(1,-1)],1)
                    write=-target_log_probs(actor,context,y[:,:8],phase='draft_writer').mean()
                    loss=.5*raw+.5*read+.1*write
        guard();loss.backward();guard()
        if initial:
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Complete finite independent gradient required')
            # The registered cold-array recovery may still be completing.
            # Keep the CPU result in memory instead of competing for its last
            # protected write allowance. This wait never admits Native work.
            while shutil.disk_usage(a.root).free<48*1024**3:
                guard();(out/'progress.json').write_text(json.dumps(dict(stage='waiting_for_registered_storage_recovery',wall_seconds=time.time()-started)))
                time.sleep(2)
            torch.save({n:p.grad for n,p in model.named_parameters()},out/'gradients.pt')
            result.update(status='PASS_DISCOVERY_INITIAL_ORDINARY_CPU_REFERENCE',gradient_relative=identity+'/gradients.pt',gradient_sha256=file_sha256(out/'gradients.pt'))
        else:
            gradients=[torch.load(native/f'rank{k}-mature-gradients.pt',weights_only=True,mmap=True) for k in range(64)]
            errors={};diff_total=ref_total=0.
            for name,p in model.named_parameters():
                difference=reference=0.
                for rank in range(64):
                    wanted=p.grad.chunk(64,0)[rank].double();actual=gradients[rank][name].double()
                    difference+=float((actual-wanted).square().sum());reference+=float(wanted.square().sum())
                errors[name]=(difference/max(reference,1e-30))**.5;diff_total+=difference;ref_total+=reference
            result.update(parameters_checked=len(errors),gradient_relative_l2=(diff_total/max(ref_total,1e-30))**.5,
                largest_named_relative_l2=max(errors.values()),worst_named_parameter=max(errors,key=errors.get))
            if max(errors.values())>.02:raise AssertionError('Mature full gradient exceeds unchanged2% per named parameter')
            result['status']='PASS_WORKSPACE_DISCOVERY_MATURE_CPU_AUDIT'
        result['joint_loss']=float(loss.detach())
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path)
    p.add_argument('--arm',choices=['initial-ordinary','ordinary','workspace','pause','draft'],required=True);run(p.parse_args())
