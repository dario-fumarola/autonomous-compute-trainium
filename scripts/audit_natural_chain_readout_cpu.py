"""Independent full-panel FP32 original-answer loss for the read-only baseline."""
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
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--case',required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time();a.output.mkdir(parents=True);result=dict(status='STOPPED_INDEPENDENT_ORIGINAL_CHAIN_READOUT')
    try:
        torch.set_num_threads(12);torch.set_num_interop_threads(1)
        if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Readout plan changed')
        plan=json.loads(a.plan.read_text());case=plan['cases'][a.case];upstream=a.root/'upstream-parcae-69284c1'
        if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned upstream reference required')
        sys.path.insert(0,str(upstream));paths=[(a.root/case['checkpoint'],case['checkpoint_sha256']),(a.root/plan['panel'],plan['panel_sha256']),(a.root/plan['noise'],plan['noise_sha256'])]
        if any(file_sha256(p)!=h for p,h in paths):raise AssertionError('Readout asset changed')
        native=json.loads((a.capture/'report.json').read_text())
        if native['status']!='COMPLETE_READONLY_ORIGINAL_CHAIN_BASELINE' or native['plan_sha256']!=a.plan_sha or native['case']!=a.case or file_sha256(a.capture/'records.json')!=native['records_sha256']:raise ValueError('Complete immutable same-case readout required')
        records=json.loads((a.capture/'records.json').read_text());observed={(r['document_id'],r['depth']):r for r in records}
        saved=torch.load(paths[0][0],map_location='cpu',mmap=True,weights_only=False);config=ParcaeCandidateConfig(**plan['model']);model=make_reference(config,True).eval();model.load_state_dict({k:v for k,v in saved['model'].items() if not k.startswith('future_lm_head.') and k!='future_positions'},strict=True);del saved
        current=None;model.initialize_state=lambda _:current;panel=json.loads(paths[1][0].read_text());noise=np.load(paths[2][0],mmap_mode='r');totals={n:[0.,0,0.] for n in case['depths']}
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for i,row in enumerate(panel):
                free=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if time.time()-started>600 or free<512*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Readout audit time/resource cap')
                view=row['views'][0];prefix=view['prefix_ids'];answer=view['answer_ids'];sequence=prefix+answer+[2];sequence+=[2]*(129-len(sequence));inputs=torch.tensor(sequence[:-1])[None];targets=torch.tensor(sequence[1:])[None];lo=len(prefix)-1;hi=lo+len(answer);current=torch.from_numpy(np.array(noise[i:i+1],copy=True))
                for n in case['depths']:
                    logits=model(inputs,return_logits=True,num_steps_pair=torch.tensor([0,n]))['logits'];terms=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(targets);value=float(terms[:,lo:hi].sum())
                    totals[n][0]+=value;totals[n][1]+=len(answer);totals[n][2]=max(totals[n][2],abs(value-observed[(row['id'],n)]['nll_sum'])/len(answer))
        scores={str(n):dict(answer_nll=v[0]/v[1],native_minus_cpu=native['depths'][str(n)]['answer_nll']-v[0]/v[1],maximum_document_mean_difference=v[2],answer_tokens=v[1]) for n,v in totals.items()}
        passed=all(abs(v['native_minus_cpu'])<=.003 for v in scores.values());result=dict(status='PASS_INDEPENDENT_ORIGINAL_CHAIN_READOUT' if passed else 'FAIL_INDEPENDENT_ORIGINAL_CHAIN_READOUT',case=a.case,depths=scores,plan_sha256=a.plan_sha)
        if not passed:raise RuntimeError('Whole-panel FP32 loss agreement failed')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,scope='Complete new discovery DEV panel,independent pinned upstream FP32 forward and scalar original-answer target positions. Verifies teacher-forced answer NLL,not exact identity of every greedy argmax.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
