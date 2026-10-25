"""Common consumed natural DEV retention after the bounded curriculum pilot."""
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
from torch.nn.attention import SDPBackend,sdpa_kernel
from compute_trainium.data import document_windows
from compute_trainium.future_span_evaluation import window_state
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--case',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time();a.output.mkdir(parents=True);result=dict(status='STOPPED_CHAIN_PILOT_RETENTION')
    try:
        torch.set_num_threads(12);torch.set_num_interop_threads(1)
        if file_sha256(a.plan)!=a.plan_sha or file_sha256(a.checkpoint)!=a.checkpoint_sha:raise AssertionError('Frozen pilot/retention checkpoint changed')
        plan=json.loads(a.plan.read_text());case=plan['cases'][a.case];source=a.root/plan['retention']['corpus']
        if file_sha256(source)!=plan['retention']['corpus_sha256']:raise AssertionError('Consumed natural DEV changed')
        upstream=a.root/'upstream-parcae-69284c1'
        if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned retention model required')
        sys.path.insert(0,str(upstream));saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if saved['purpose']!=plan['purpose'] or saved['case']!=a.case or saved['update']!=case['total_updates'] or saved['qualification_only']:raise ValueError('Completed registered scientific pilot endpoint required')
        if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=saved['state_digest']:raise AssertionError('Full scientific state readback differs')
        cfg=ParcaeCandidateConfig(**plan['model']);model=make_reference(cfg,True).eval();model.load_state_dict(saved['model'],strict=True);del saved;initial=digest(dict(sorted(model.state_dict().items())))
        current=None;model.initialize_state=lambda _:current;depth=1 if case['arm']=='one' else 2;scores=[];windows=0
        docs=sorted([d for d in json.loads(source.read_text()) if d['split']=='dev' and d['cohort']=='curve'],key=lambda d:d['id'])
        if not docs:raise ValueError('Declared retention cohort empty')
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            for index,d in enumerate(docs):
                free=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if time.time()-started>1200 or free<512*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Retention CPU cap/reserve')
                total=0.;count=0
                for window,(tokens,mask) in enumerate(document_windows(d['tokens'],1024)):
                    n=int(mask.sum());current=window_state(cfg,window,plan['noise_seed'],n);logits=model(tokens[None,:n],return_logits=True,num_steps_pair=torch.tensor([depth-1,1]))['logits'][0]
                    for lo in range(0,n,128):
                        hi=min(n,lo+128);values=logits[lo:hi].double();target=tokens[lo+1:hi+1];total+=float((values.logsumexp(-1)-values.gather(-1,target[:,None]).squeeze(-1)).sum())
                    count+=n;windows+=1
                if count!=len(d['tokens']):raise AssertionError('Retention discarded original targets')
                scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
                if (index+1)%100==0:write_state(a.output/'progress.json',dict(documents=index+1,windows=windows,elapsed_seconds=time.time()-started))
        if digest(dict(sorted(model.state_dict().items())))!=initial:raise AssertionError('Retention scoring changed parameters')
        write_state(a.output/'document-scores.json',dict(records=[dict(id=s.document,group=s.group,domain=s.domain,nll_sum=s.nll_sum,tokens=s.tokens) for s in scores]))
        result=dict(status='COMPLETE_CHAIN_PILOT_RETENTION',case=a.case,depth=depth,metrics=summarize(scores),documents=len(scores),windows=windows,model_unchanged=True,checkpoint_sha256=a.checkpoint_sha,plan_sha256=a.plan_sha)
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,scope='Whole consumed natural DEV curve cohort,independent upstream FP32,all original source targets and60/20/20 domain weights. Retention readout,not fresh confirmation.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
