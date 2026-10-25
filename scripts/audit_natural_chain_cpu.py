"""Independent upstream FP32 ordered-conclusion gradient and optimizer check."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from compute_trainium.future_span_reference_optimizer import reference_optimizer,fp32_reference_execution
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_recovery_check import moments
from compute_trainium.parcae_runtime_scalars import changing_qualification_schedule
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.parcae_update_qualification import named_moments
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest
from scripts.run_parcae_optimizer_native import step_comparison,step_passes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--update',type=int,choices=[0,120],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time();a.output.mkdir(parents=True)
    result=dict(status='STOPPED_INDEPENDENT_NATURAL_CHAIN_AUDIT')
    def guard():
        free=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if time.time()-started>600 or free<512*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Independent task CPU audit resource cap')
    try:
        torch.set_num_threads(12);torch.set_num_interop_threads(1);guard()
        if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Task plan changed')
        plan=json.loads(a.plan.read_text());upstream=a.root/'upstream-parcae-69284c1'
        if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned upstream required')
        sys.path.insert(0,str(upstream))
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):raise AssertionError('Frozen task source changed')
        native=json.loads((a.capture/'report.json').read_text());point=next(x for x in native['probes'] if x['update']==a.update)
        if native['status']!='CAPTURED_NATURAL_CHAIN_DRIVER_QUALIFICATION' or native['signature']['plan_sha256']!=a.plan_sha:raise ValueError('Complete same-plan task capture required')
        base=a.root/plan['identity'];tasks_file=base/'tasks.pt'
        if file_sha256(tasks_file)!=plan['tasks_sha256']:raise AssertionError('Original task assets changed')
        tasks=torch.load(tasks_file,map_location='cpu',mmap=True,weights_only=False);stage=0 if a.update<40 else 2
        state_file=base/'initial-body.pt' if a.update==0 else a.capture/'full-state-u120.pt'
        expected=plan['initial_sha256'] if a.update==0 else native['full_checkpoint_sha256']
        if file_sha256(state_file)!=expected or file_sha256(a.capture/point['gradient_file'])!=point['gradient_sha256']:raise AssertionError('Task state/vector changed')
        saved=torch.load(state_file,map_location='cpu',mmap=True,weights_only=False)
        if a.update==120 and digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=native['full_checkpoint_state_digest']:raise AssertionError('Full task state digest differs')
        cfg=ParcaeCandidateConfig(**plan['model']);model=make_reference(cfg,True);model.load_state_dict(saved['model'],strict=True)
        current=None;model.initialize_state=lambda _:current;model_digest=digest(dict(sorted(model.state_dict().items())))
        # Independent scalar construction of answer/EOS positions from lengths;
        # no native objective or shared mask/loss helper is used.
        answer_count=sum(len(row['answer_ids']) for row in tasks['metadata'][stage]);eos_count=32
        if answer_count!=point['answer_tokens'] or point['stage']!=stage:raise AssertionError('Global task denominator differs')
        n=1 if native['case']=='one' else 4
        if point['n']!=n:raise AssertionError('Qualification supported depth differs')
        total_answer=total_stop=0.
        with sdpa_kernel(SDPBackend.MATH):
            for rank,row in enumerate(tasks['metadata'][stage]):
                guard();prefix=row['prefix_ids'];answer=row['answer_ids'];sequence=prefix+answer+[2];pad=129-len(sequence)
                if pad<0:raise AssertionError('Task exceeds fixed context')
                sequence += [2]*pad;inputs=torch.tensor(sequence[:-1])[None];targets=torch.tensor(sequence[1:])[None]
                if not torch.equal(inputs,tasks['inputs'][stage,rank:rank+1]) or not torch.equal(targets,tasks['targets'][stage,rank:rank+1]):raise AssertionError('Independent task input/target assembly differs')
                first=len(prefix)-1;last=first+len(answer);mask=torch.zeros(1,128,dtype=torch.bool);mask[:,first:last]=True
                stop=torch.zeros_like(mask);stop[:,last]=True
                if not torch.equal(mask,tasks['answer_mask'][stage,rank:rank+1]) or not torch.equal(stop,tasks['termination_mask'][stage,rank:rank+1]):raise AssertionError('Independent original-answer/EOS positions differ')
                current=tasks['h0'][rank:rank+1].clone();logits=model(inputs,return_logits=True,num_steps_pair=torch.tensor([0,n]))['logits']
                terms=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(targets)
                loss_answer=terms[:,first:last].sum();loss_stop=terms[:,last].sum()
                (loss_answer/answer_count+.1*loss_stop/eos_count).backward();total_answer+=float(loss_answer.detach());total_stop+=float(loss_stop.detach())
        if digest(dict(sorted(model.state_dict().items())))!=model_digest:raise AssertionError('CPU backward changed task weights')
        vector=torch.load(a.capture/point['gradient_file'],map_location='cpu',mmap=True,weights_only=False)
        delta=gradient_comparison({k:v.grad.detach() for k,v in model.named_parameters()},vector['gradients'])
        differences=dict(answer=float(vector['parts'][0])/answer_count-total_answer/answer_count,eos=float(vector['parts'][1])/32-total_stop/32)
        actual_file=a.capture/('after-first-update.pt' if a.update==0 else 'after121.pt');expected_sha=native['first_update_sha256'] if a.update==0 else native['final_sha256']
        if file_sha256(actual_file)!=expected_sha:raise AssertionError('Actual task optimizer step changed')
        actual=torch.load(actual_file,map_location='cpu',mmap=True,weights_only=False)
        for name,parameter in model.named_parameters():parameter.grad=vector['gradients'][name].clone()
        optimizer=reference_optimizer(model,actual['optimizer']['recipe']['groups'],saved.get('optimizer'))
        before={name:p.detach().clone() for name,p in model.named_parameters()};norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False));changing_qualification_schedule(optimizer,a.update)
        with fp32_reference_execution():optimizer.step()
        left=dict(model={k:p.detach() for k,p in model.named_parameters()},moments=named_moments(model,optimizer,reference=True))
        right=dict(model={k:actual['model'][k] for k in before},moments=moments(actual['optimizer']))
        step=step_comparison(before,left,before,right);ok=delta['passes_two_percent_gate'] and all(abs(v)<=.003 for v in differences.values()) and step_passes(step)
        result=dict(status='PASS_INDEPENDENT_NATURAL_CHAIN_AUDIT' if ok else 'FAIL_INDEPENDENT_NATURAL_CHAIN_AUDIT',case=native['case'],mode=native['mode'],update=a.update,examples=32,n=n,gradient=delta,loss_differences=differences,optimizer_step=step,optimizer_step_passes_two_percent_gate=step_passes(step),cpu_raw_gradient_norm=norm,plan_sha256=a.plan_sha)
        if not ok:raise RuntimeError('Task numerical/optimizer gate failed')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(wall_seconds=time.time()-started,assigned_chips=0,research_optimizer_updates=0,test_accessed=False,scope='Independent pinned upstream FP32 model and independent scalar original-answer/EOS masks;all32 TRAIN tasks and independent observed-gradient optimizer step. No held-out result or curriculum efficacy claim.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
