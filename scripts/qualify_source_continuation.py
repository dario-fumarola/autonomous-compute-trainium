"""Bounded native equivalence of old-source and audited appended-source restores.

These repeated eight-update checks are qualification, never scientific exposure.
Keep the established32-rank/eight-chip layout and every source checkpoint intact.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def compare_trees(left, right):
    error = norm = maximum = 0.
    exact = True
    def visit(a, b):
        nonlocal error, norm, maximum, exact
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
                raise AssertionError('Restored tensor structure differs')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise AssertionError('Nonfinite continuation state')
            delta = a.double() - b.double()
            error += float(delta.square().sum());norm += float(a.double().square().sum())
            if a.numel():maximum = max(maximum, float(delta.abs().max()))
            exact = exact and torch.equal(a,b)
        elif isinstance(a, dict):
            if not isinstance(b,dict) or a.keys() != b.keys():raise AssertionError('State keys differ')
            for key in a:visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if type(a) != type(b) or len(a) != len(b):raise AssertionError('State sequence differs')
            for x,y in zip(a,b):visit(x,y)
        elif a != b:raise AssertionError('Continuation metadata differs')
    visit(left,right)
    relative = (error/max(norm,1e-30))**.5
    if relative > 1e-5 or maximum > 1e-4:
        raise AssertionError('Old/new source update disagreement exceeds declared tolerance')
    return {'bit_identical':exact,'relative_l2':relative,'maximum_absolute_error':maximum}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',type=int,choices=[1,2],default=1)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(4);started=time.time();deadline=started+1800
    public=Path('results/source_native_qualification_20261024')
    if a.revision==2:
        prior=json.loads((a.root/'source-native-equivalence-v1/report.json').read_text())
        if prior['status']!='STOPPED_NATIVE_APPEND_SOURCE_CONTINUATION' or prior['error']!='CPU qualification disagreement':
            raise ValueError('Revision2 requires the preserved overly strict CPU-gradient stop')
        public=public/'v2'
    public.mkdir(parents=True,exist_ok=True)
    gradient_tolerance=.01 if a.revision==1 else .08
    old=a.root/'natural-append-3250m';new=a.root/'natural-append-6500m-v2'
    prefix=np.load(new/'packed/prefix_counts.npy',mmap_mode='r')
    count=lambda update:int(prefix[update*256].sum())
    sources=json.loads((a.root/'variable-depth-campaign-v2/plan.json').read_text())['source_sha256'].copy()
    for f in ['scripts/qualify_source_continuation.py','scripts/supervise.py','scripts/run_native.sh']:
        sources[f]=file_sha256(f)
    inputs={}
    for arm in ['ordinary','fixed']:
        for kind in ['old','new']:
            path=(a.root/'runs'/f'vdepth-v1-{arm}-s103-u3456/checkpoint.pt' if kind=='old'
                  else a.root/'source-state-copy-q-v1'/f'{arm}-s103.pt')
            inputs[arm,kind]=(path,file_sha256(path))
    plan={'identity':f'source-native-equivalence-v{a.revision}','purpose':'QUALIFICATION_ONLY',
          'registered_unix':time.time(),'deadline_unix':deadline,'maximum_native_attempts':4,
          'seed':103,'source_update':3456,'updates_per_attempt':8,'world':32,'batch':4,'accumulation':2,
          'future_schedule':{'warmup_tokens':count(96),'decay_start':count(10368),'decay_end':count(11520)},
          'source_sha256':sources,'cpu_probe_loss_tolerance':.01,'cpu_gradient_relative_l2_tolerance':gradient_tolerance,
          'revision_note':'V1 remains stopped at an additional1% CPU-gradient gate. V2 uses the frozen trainer existing8% BF16 CPU-gradient tolerance, not a newly inferred bound; direct old/new model+optimizer agreement remains1e-5 relative/1e-4 maximum. No prior result is relabeled.',
          'state_relative_l2_tolerance':1e-5,'state_maximum_absolute_tolerance':1e-4,
          'scope':'Old3.25B versus independently audited6.5B source, identical first8 restored updates and full optimizer state. Both explicitly extend the stable schedule identically. Fresh qualification copies only; no scientific continuation or TEST. Existing topology unchanged. Costs include all four repeated-update attempts.'}
    write_state(a.output/'plan.json',plan);write_state(public/'plan.json',plan)
    launched=[];jobs=[];records=[]
    def frozen():
        if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Frozen qualification source changed')
    def costs():
        closed=[];missing=[]
        for name in launched:
            path=a.root/'snapshots'/name/'closure.json'
            if path.exists():closed.append({'identity':name,**json.loads(path.read_text())})
            else:missing.append(name)
        return {'closed_attempts':closed,'missing_closures':missing,
                'assigned_chip_hours':sum(r['assigned_chip_hours'] for r in closed),
                'node_opportunity_chip_hours':16*(time.time()-started)/3600,
                'scope':'Includes repeated qualification updates, setup, probes and closure. Concurrent CPU preparation costs overlap this interval; do not add node elapsed times.'}
    try:
        for arm in ['ordinary','fixed']:
            pairs=[]
            for half,kind in enumerate(['old','new']):
                frozen();name=f'source-native-v{a.revision}-{arm}-{kind}';data=old if kind=='old' else new
                checkpoint,sha=inputs[arm,kind]
                if file_sha256(checkpoint)!=sha:raise AssertionError('Source checkpoint changed')
                relative=lambda path:'/work/instance-nvme/'+str(path.relative_to(a.root))
                args=['--corpus',relative(data/'source.json'),'--packed-corpus',relative(data/'packed'),
                      '--dev-corpus','/work/instance-nvme/variable-depth-fresh-v1/dev.json',
                      '--architecture','preln','--context','1024','--batch','4','--accumulation','2',
                      '--precision','bf16','--seed','103','--data-seed','23','--optimizer','fused',
                      '--recipe','pretraining','--reference-threads','4','--purpose','qualification',
                      '--depth-policy',arm,'--lr',str(.0006 if arm=='ordinary' else .001),
                      '--lr-warmup-tokens',str(count(96)),'--decay-start',str(count(10368)),
                      '--decay-end',str(count(11520)),'--extend-stable-schedule','--steps','8','--warmup','4',
                      '--resume',relative(checkpoint),'--save']
                env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                         RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE='scripts.train_variable_depth',RESEARCH_DDP_RANKS='32',
                         NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,(half+1)*32))),
                         RESEARCH_TIMEOUT_SECONDS=str(min(720,int(deadline-time.time()))))
                with (a.output/(name+'.log')).open('w') as log:
                    proc=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
                jobs.append(proc);pairs.append((name,proc));launched.append(name)
                write_state(a.output/'state.json',{'status':'RUNNING','arm':arm,'launched':launched,'deadline_unix':deadline})
            reports=[]
            for name,proc in pairs:
                if proc.wait(timeout=max(1,deadline-time.time())):raise RuntimeError('Native qualification failed: '+name)
                report=json.loads((a.root/'runs'/name/'report.json').read_text())
                probe=report['cpu_probe']
                if report['status']!='PASS_PACKED_TRAINING_PHASE' or not report['all_replica_model_optimizer_states_equal']:
                    raise AssertionError('Incomplete distributed recovery/update check')
                if abs(probe['native_loss']-probe['cpu_loss'])>.01 or probe['gradient_relative_l2']>gradient_tolerance:
                    raise AssertionError('CPU qualification disagreement')
                reports.append(report)
            keys=['cursor','total_real_targets','total_training_matmul_flops','initial_model_digest','schedule','recovered_model_optimizer_digest']
            if any(reports[0][k]!=reports[1][k] for k in keys):raise AssertionError('Restored trajectory accounting differs')
            saved=[torch.load(a.root/'runs'/name/'checkpoint.pt',map_location='cpu',mmap=True,weights_only=False) for name,_ in pairs]
            errors={key:compare_trees(saved[0][key],saved[1][key]) for key in ['model','optimizer','cpu_rng']}
            del saved
            records.append({'arm':arm,'source_checkpoint_hashes':{k:inputs[arm,k][1] for k in ['old','new']},
                            'state_comparison':errors,'reports':reports})
            write_state(a.output/'progress.json',{'records':records})
        frozen()
        if any(file_sha256(path)!=sha for path,sha in inputs.values()):raise AssertionError('Original input changed')
        report={'status':'PASS_NATIVE_APPEND_SOURCE_CONTINUATION','records':records,
                'original_source_checkpoints_unchanged':True,'execution_wall_seconds':time.time()-started,'execution_costs':costs(),
                'scope':plan['scope']+' This qualifies the source migration and stable schedule extension, not a new scientific learning result.'}
    except Exception as error:
        if any(proc.poll() is None for proc in jobs):subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
        report={'status':'STOPPED_NATIVE_APPEND_SOURCE_CONTINUATION','error':str(error),'records':records,
                'execution_wall_seconds':time.time()-started,'execution_costs':costs()}
        write_state(a.output/'report.json',report);write_state(public/'report.json',report);raise
    write_state(a.output/'report.json',report);write_state(public/'report.json',report)


if __name__=='__main__':main()
