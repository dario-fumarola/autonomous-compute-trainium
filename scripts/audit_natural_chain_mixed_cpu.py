"""Independent complete mixed-batch gradient and actual first optimizer update."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from compute_trainium.future_span_reference_optimizer import reference_optimizer,fp32_reference_execution
from compute_trainium.natural_chain_mixed import optimizer_schedule
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_recovery_check import moments
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.parcae_update_qualification import named_moments
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.run_parcae_optimizer_native import step_comparison,step_passes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--case',required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time();a.output.mkdir(parents=True);result=dict(status='STOPPED_INDEPENDENT_MIXED_PILOT_INTEGRATION')
    def guard():
        free=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if time.time()-started>1200 or free<512*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Mixed CPU integration cap/reserve')
    try:
        torch.set_num_threads(12);torch.set_num_interop_threads(1);guard()
        if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen mixed plan changed')
        plan=json.loads(a.plan.read_text());case=plan['cases'][a.case];upstream=a.root/'upstream-parcae-69284c1'
        if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned upstream reference required')
        sys.path.insert(0,str(upstream))
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):raise AssertionError('Frozen mixed source changed')
        native=json.loads((a.capture/'report.json').read_text())
        if native['status']!='CAPTURED_MIXED_PILOT_INTEGRATION' or native['case']!=a.case or native['plan_sha256']!=a.plan_sha or native['update']!=1:raise ValueError('Exact complete first mixed update required')
        point=native['probe'];base=a.root/plan['identity']
        for file,expected in [(base/'initial-body.pt',plan['initial_sha256']),(base/'tasks.pt',plan['tasks_sha256']),(a.capture/'gradient-u0.pt',point['gradient_sha256']),(a.capture/native['checkpoint_file'],native['checkpoint_sha256'])]:
            if file_sha256(file)!=expected:raise AssertionError('Mixed integration asset changed')
        saved=torch.load(base/'initial-body.pt',map_location='cpu',mmap=True,weights_only=False);cfg=ParcaeCandidateConfig(**plan['model']);model=make_reference(cfg,True);model.load_state_dict(saved['model'],strict=True);del saved
        current=None;model.initialize_state=lambda _:current;before_digest=digest(dict(sorted(model.state_dict().items())))
        packed=PackedCorpus.from_mapped(a.root/plan['data']['packed'],1024,23,plan['data']['corpus_sha256']);tasks=torch.load(base/'tasks.pt',map_location='cpu',mmap=True,weights_only=False)
        cursor=plan['ntp_start_cursor'];ntp_count=sum(packed.counts(cursor,128));problem_ids=tasks['order'][0];view_ids=tasks['view_indices'][0,problem_ids]
        count=sum(len(tasks['metadata'][int(i)]['answer_ids']) for i in view_ids) if case['task_weight'] else 0;n=1 if case['arm']=='one' else 4
        if (point['ntp_cursor'],point['ntp_count'],point['answer_count'],point['n'])!=(cursor,ntp_count,count,n):raise AssertionError('Independent whole-batch denominators/depth differ')
        noise={}
        for kind in ['ntp']+(['task'] if case['task_weight'] else []):
            arrays=[]
            for rank,h in enumerate(point['noise_'+kind+'_sha256']):
                f=a.capture/f'h0-{kind}-rank{rank}.npy'
                if file_sha256(f)!=h:raise AssertionError('Captured initial latent states changed')
                arrays.append(np.load(f,mmap_mode='r'))
            noise[kind]=arrays
        sums=[0.,0.,0.]
        with sdpa_kernel(SDPBackend.MATH):
            for kind in ['ntp']+(['task'] if case['task_weight'] else []):
                for slot in range(128):
                    guard();rank=slot%32;micro=slot//32;current=torch.from_numpy(np.array(noise[kind][rank][micro:micro+1],copy=True))
                    if kind=='ntp':
                        batch=packed.batch([cursor+slot],0)[0];inputs,targets,mask=batch.inputs,batch.targets,batch.target_mask
                    else:
                        row=tasks['metadata'][int(view_ids[slot])];prefix=row['prefix_ids'];answer=row['answer_ids'];sequence=prefix+answer+[2];sequence += [2]*(129-len(sequence));inputs=torch.tensor(sequence[:-1])[None];targets=torch.tensor(sequence[1:])[None];lo=len(prefix)-1;hi=lo+len(answer)
                        expected=int(view_ids[slot])
                        if not torch.equal(inputs,tasks['inputs'][expected:expected+1]) or not torch.equal(targets,tasks['targets'][expected:expected+1]):raise AssertionError('Independent task assembly differs')
                    logits=model(inputs,return_logits=True,num_steps_pair=torch.tensor([0,n]))['logits'];terms=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(targets)
                    if kind=='ntp':value=(terms*mask).sum();(value/ntp_count).backward();sums[0]+=float(value.detach())
                    else:
                        answer=terms[:,lo:hi].sum();stop=terms[:,hi].sum();(case['task_weight']*answer/count+case['task_weight']*.1*stop/128).backward();sums[1]+=float(answer.detach());sums[2]+=float(stop.detach())
                    if (slot+1)%32==0:write_state(a.output/'progress.json',dict(branch=kind,examples=slot+1,elapsed_seconds=time.time()-started))
        if digest(dict(sorted(model.state_dict().items())))!=before_digest:raise AssertionError('Independent backward changed weights')
        vector=torch.load(a.capture/'gradient-u0.pt',map_location='cpu',mmap=True,weights_only=False);delta=gradient_comparison({k:v.grad.detach() for k,v in model.named_parameters()},vector['gradients'])
        denominators=[ntp_count,max(1,count),128];differences={name:(float(vector['parts'][i])-sums[i])/denominators[i] for i,name in enumerate(['ntp','original_answer','task_eos'])}
        actual=torch.load(a.capture/native['checkpoint_file'],map_location='cpu',mmap=True,weights_only=False)
        if digest({'model':dict(sorted(actual['model'].items())),'optimizer':actual['optimizer']})!=native['full_state_digest']:raise AssertionError('Full first update state changed')
        optimizer=reference_optimizer(model,actual['optimizer']['recipe']['groups'])
        for name,p in model.named_parameters():p.grad=vector['gradients'][name].clone()
        before={name:p.detach().clone() for name,p in model.named_parameters()};norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False));optimizer_schedule(optimizer,0,case['total_updates'],peak=case['peak_lr'])
        with fp32_reference_execution():optimizer.step()
        left=dict(model={k:p.detach() for k,p in model.named_parameters()},moments=named_moments(model,optimizer,reference=True));right=dict(model={k:actual['model'][k] for k in before},moments=moments(actual['optimizer']));step=step_comparison(before,left,before,right)
        passed=delta['passes_two_percent_gate'] and all(abs(v)<=.003 for v in differences.values()) and step_passes(step)
        result=dict(status='PASS_INDEPENDENT_MIXED_PILOT_INTEGRATION' if passed else 'FAIL_INDEPENDENT_MIXED_PILOT_INTEGRATION',case=a.case,examples=dict(ntp=128,tasks=128 if case['task_weight'] else 0),gradient=delta,loss_differences=differences,optimizer_step=step,optimizer_step_passes_two_percent_gate=step_passes(step),cpu_raw_gradient_norm=norm,plan_sha256=a.plan_sha)
        if not passed:raise RuntimeError('Independent mixed gradient/loss/optimizer gate failed')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,prescribed_cpu_optimizer_steps=1,research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,scope='Independent pinned upstream FP32 mixed global objective,all128 original NTP windows and applicable128 ordered source conclusions;independent target positions and observed-gradient optimizer step. No held-out quality readout.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
