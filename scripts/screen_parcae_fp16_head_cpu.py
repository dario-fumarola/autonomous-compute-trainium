"""Finite TRAIN-only head-precision feasibility at mature fitted checkpoints."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_fp16_head_candidate import ParcaeFP16HeadCandidate
from compute_trainium.parcae_training_candidate import gradient_comparison,initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/'parcae-fp16-head-cpu-v2';public=Path('results/parcae_fp16_head_cpu_20261025/v2')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    parent_file=a.root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json';parent=json.loads(parent_file.read_text())
    points=[]
    for arm,depth in [('fixed1',1),('poisson2',2)]:
        identity='low-depth-main-r2-'+arm+'-finish';report=json.loads((a.root/'runs'/identity/'report.json').read_text())
        row=next(r for r in report['checkpoint_records'] if r['update']==report['update'])
        if (report['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or report['qualification_only']
                or report['plan_sha256']!=file_sha256(parent_file) or not row['all_complete_replica_states_equal']):
            raise ValueError('Completed same-plan all-replica fitted checkpoint required')
        points.append({'arm':arm,'depth':depth,'backward_cap':depth,'case_id':report['case_id'],
            'checkpoint':f'runs/{identity}/checkpoint-u{report["update"]}.pt','checkpoint_sha256':row['checkpoint_sha256'],
            'full_state_digest':row['full_state_digest'],'training_update':report['update'],'source_cursor':report['source_cursor']})
    source=dict(parent['source_sha256'])
    for path in ['compute_trainium/parcae_fp16_head_candidate.py','scripts/screen_parcae_fp16_head_cpu.py',
                 'scripts/audit_parcae_precise_training_point.py','scripts/qualify_parcae_candidate_cpu.py',
                 'tests/test_parcae_fp16_head_candidate.py']:
        source[path]=file_sha256(path)
    plan={'identity':'parcae-fp16-head-cpu-v2','registered_unix':started,'parent_plan_sha256':file_sha256(parent_file),
        'source_sha256':source,'points':points,'windows_per_point':8,'seconds_limit':600,
        'gradient_relative_gate':.02,'loss_absolute_gate':.003,'loss_scale':8192,'upstream_commit':PIN,
        'precisions':['independent_cpu_fp32','cpu_fp16_body_fp32_head','cpu_fp16_body_fp16_head'],
        'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
        'scope':'Prospective performance-path feasibility on first8 next natural TRAIN windows at two mature checkpoints. Fixed supported depths1/2,full gradients common to every precision. FP32 masters/logits/loss and manual8192 scaling for mixed paths. No held-out score,native operator/update/recovery qualification,speed claim or change to any frozen study.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result={'status':'STOPPED_CPU_FP16_HEAD_SCREEN'};records=[];comparisons=[];work=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time()-started>plan['seconds_limit'] or available<544*1024**3
                or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3):
            raise RuntimeError('CPU projection feasibility reserve/cap')
    try:
        guard();torch.set_num_threads(8);torch.set_num_interop_threads(1)
        upstream=a.root/'upstream-parcae-69284c1'
        if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned upstream FP32 reference required')
        sys.path.insert(0,str(upstream));config=ParcaeCandidateConfig(**parent['model'])
        packed=PackedCorpus.from_mapped(a.root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
        if packed.fingerprint!=parent['data']['packing_fingerprint']:raise AssertionError('Frozen natural packing changed')
        for point in points:
            guard();path=a.root/point['checkpoint']
            if file_sha256(path)!=point['checkpoint_sha256']:raise AssertionError('Mature checkpoint changed')
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=point['full_state_digest']:
                raise AssertionError('Complete checkpoint readback differs')
            if saved['case_id']!=point['case_id'] or saved['cursor']!=point['source_cursor'] or saved['qualification_only']:
                raise AssertionError('Fitted checkpoint identity/cursor differs')
            windows=range(point['source_cursor'],point['source_cursor']+8);total=sum(packed.counts(point['source_cursor'],8))
            case=parent['cases'][point['case_id']];gold=None;gold_nll=None;point_rows=[]
            for mode in plan['precisions']:
                guard();begin=time.time();mixed=mode!='independent_cpu_fp32'
                cls=ParcaeFP16HeadCandidate if mode.endswith('body_fp16_head') else ParcaeFP16Candidate
                model=cls(config,initialize=False) if mixed else make_reference(config,True)
                model.load_state_dict(saved['model'],strict=True);before=digest(dict(sorted(model.state_dict().items())))
                current=None
                if not mixed:model.initialize_state=lambda _:current
                scale=8192 if mixed else 1;loss=0.
                with sdpa_kernel(SDPBackend.MATH):
                    for window in windows:
                        guard();batch=packed.batch([window],0)[0];current=initial_state(config,window,case['state_seed']);n=point['depth']
                        logits=(model(batch.inputs,current,total_steps=n,backward_cap=n) if mixed else
                            model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([0,n]))['logits'])
                        if logits.dtype!=torch.float32:raise AssertionError('Loss must receive FP32 logits')
                        value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
                        (value*(scale/total)).backward();loss+=float(value.detach())
                        work+=matmul_ledger(config,1024,total_steps=n,backward_cap=n)['training_matmul_flops']
                vector={name:p.grad.detach().clone()/scale for name,p in model.named_parameters()}
                if any(not torch.isfinite(g).all() for g in vector.values()):raise FloatingPointError('Nonfinite scaled projection gradient')
                if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('CPU screen changed parameters')
                if gold is None:gold=vector;gold_nll=loss/total
                comparison=gradient_comparison(gold,vector);difference=loss/total-gold_nll
                row={'arm':point['arm'],'training_update':point['training_update'],'depth':point['depth'],
                    'precision':mode,'nll':loss/total,'loss_difference_from_fp32':difference,'gradient':comparison,
                    'real_train_targets':total,'parameters_unchanged':True,'wall_seconds':time.time()-begin,
                    'passes_small_cpu_gate':comparison['passes_two_percent_gate'] and abs(difference)<=.003}
                records.append(row);point_rows.append(row);write_state(public/'progress.json',{'records':records})
                del model,vector;gc.collect()
            comparisons.append({'arm':point['arm'],'all_mixed_paths_pass':all(r['passes_small_cpu_gate'] for r in point_rows[1:])})
            del saved,gold;gc.collect()
        guard()
        if any(file_sha256(p)!=sha for p,sha in source.items()):raise AssertionError('Frozen projection-screen source changed')
        passed=all(r['all_mixed_paths_pass'] for r in comparisons)
        result={'status':'PASS_SMALL_CPU_FP16_HEAD_SCREEN' if passed else 'FAIL_SMALL_CPU_FP16_HEAD_SCREEN',
            'comparisons':comparisons,'complete_checkpoint_readback_exact':True,'native_execution_qualified':False,
            'native_speed_measured':False,'scientific_training_changed':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(records=records,diagnostic_training_matrix_ops=work,wall_seconds=time.time()-started,
            assigned_chips=0,optimizer_updates=0,test_accessed=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
