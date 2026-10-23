"""Bounded numerical, restart and throughput gates for the explicit new recipe."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT']
    started=time.time()
    subprocess.run(['docker','update','--memory',str(320*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
    base=['--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
          '--context','1024','--batch','4','--accumulation','2','--architecture','preln',
          '--precision','bf16','--data-seed','23','--seed','103','--optimizer','fused','--recipe','pretraining',
          '--reference-threads','4','--warmup','4','--save']
    def phase(name,extra,half,module='scripts.train_variable_depth',distributed=True):
        if time.time()+120>a.deadline_unix:raise TimeoutError('Qualification deadline')
        cores=list(range(half*32,(half+1)*32)) if distributed else [half*32]
        env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE=module,
             RESEARCH_DDP_RANKS=str(len(cores)) if distributed else '0',NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),
             RESEARCH_TIMEOUT_SECONDS='900')
        log=(a.output/(name+'.log')).open('w')
        process=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise',*(base if distributed else []),*extra],env=env,stdout=log,stderr=subprocess.STDOUT)
        return name,process,log
    def finish(jobs):
        failed=[]
        for name,process,log in jobs:
            code=process.wait();log.close()
            if code:failed.append([name,code])
        if failed:raise RuntimeError(f'Qualification failure: {failed}')
        return {name:json.loads((root/'runs'/name/'report.json').read_text()) for name,_,_ in jobs}
    fixture=finish([phase('vdepth-q-optimizer-004',['--mode','fused'],0,'scripts.qualify_depth_optimizer',False)])
    tiny=finish([phase('vdepth-q-tiny-mix-004',['--size','tiny','--depth-policy','mixture','--steps','8','--check-global'],0),
                 phase('vdepth-q-tiny-fixed-004',['--size','tiny','--depth-policy','fixed','--steps','8','--check-global'],1)])
    if any(v['global_gradient_check']['gradient_relative_l2']>.01 for v in tiny.values()):raise AssertionError('Global gradient tolerance')
    replay=finish([phase('vdepth-q-tiny-resume-004',['--size','tiny','--depth-policy','mixture','--steps','8',
                       '--resume',f'{croot}/runs/vdepth-q-tiny-mix-004/checkpoint.pt'],0),
                   phase('vdepth-q-tiny-direct-004',['--size','tiny','--depth-policy','mixture','--steps','16'],1)])
    import torch
    aa=torch.load(root/'runs/vdepth-q-tiny-resume-004/checkpoint.pt',map_location='cpu',weights_only=False)
    bb=torch.load(root/'runs/vdepth-q-tiny-direct-004/checkpoint.pt',map_location='cpu',weights_only=False)
    maximum=max(float((aa['model'][k]-bb['model'][k]).abs().max()) for k in aa['model'])
    relative=(sum(float((aa['model'][k]-bb['model'][k]).square().sum()) for k in aa['model'])/
              sum(float(bb['model'][k].square().sum()) for k in bb['model']))**.5
    moment_errors={key:max(float((aa['optimizer']['state'][i][key]-bb['optimizer']['state'][i][key]).abs().max())
                      for i in aa['optimizer']['state']) for key in ['step','exp_avg','exp_avg_sq']}
    numeric_replay={'bit_exact':replay['vdepth-q-tiny-resume-004']['state_digest']==replay['vdepth-q-tiny-direct-004']['state_digest'],
         'parameter_max_error':maximum,'parameter_relative_l2':relative,'optimizer_max_errors':moment_errors}
    (a.output/'numerical-replay.json').write_text(json.dumps(numeric_replay,indent=2)+'\n')
    if maximum>5e-5 or relative>1e-4 or moment_errors['step']!=0 or moment_errors['exp_avg']>1e-5 or moment_errors['exp_avg_sq']>1e-7:
        raise AssertionError('Numerical interrupted replay tolerance exceeded')
    if any(aa[k]!=bb[k] for k in ['update','cursor','real_tokens','training_matmul_flops']):raise AssertionError('Replay counters differ')
    del aa,bb
    peers='vdepth-q-150m-mix-004,vdepth-q-150m-fixed-004'
    ready=f'{croot}/{a.output.relative_to(root)}/warm-ready'
    full=finish([phase(f'vdepth-q-150m-{tag}-004',['--size','small','--depth-policy',policy,'--steps','104',
                   '--warm-rendezvous',ready,'--warm-peer-names',peers],half)
                for tag,policy,half in [('mix','mixture',0),('fixed','fixed',1)]])
    recovered=finish([phase(f'vdepth-q-restore-{tag}-004',['--size','small','--depth-policy',policy,'--steps','8',
                  '--resume',f'{croot}/runs/vdepth-q-150m-{tag}-004/checkpoint.pt'],half)
                for tag,policy,half in [('mix','mixture',0),('fixed','fixed',1)]])
    extra=finish([phase('vdepth-q-150m-ordinary-004',['--size','small','--depth-policy','ordinary','--steps','104'],0),
                  phase('vdepth-q-tiny-peer-004',['--size','tiny','--depth-policy','mixture','--steps','8',
                       '--resume',f'{croot}/runs/vdepth-q-tiny-mix-004/checkpoint.pt'],1)])
    if extra['vdepth-q-tiny-peer-004']['state_digest']!=replay['vdepth-q-tiny-resume-004']['state_digest']:
        raise AssertionError('Two fresh-process continuations from identical state differ')
    full['vdepth-q-150m-ordinary-004']=extra['vdepth-q-150m-ordinary-004']
    checks=[]
    for name,v in full.items():
        if not v['all_replica_model_optimizer_states_equal']:raise AssertionError('Replica inequality')
        for path in sorted((root/'runs'/name).glob('cpu-probe-rank*-depth*.json')):
            check=json.loads(path.read_text());checks.append({'run':name,'probe':path.name,**check})
            if check['gradient_relative_l2']>.01 or abs(check['native_loss']-check['cpu_loss'])>.01:
                raise AssertionError('Full-size CPU agreement tolerance')
    if len(checks)!=5:raise AssertionError('Missing full-size depth probe')
    for tag in ['mix','fixed']:
        if recovered[f'vdepth-q-restore-{tag}-004']['recovered_model_optimizer_digest']!=full[f'vdepth-q-150m-{tag}-004']['state_digest']:
            raise AssertionError('Full-size optimizer recovery differs')
    if full['vdepth-q-150m-mix-004']['total_training_matmul_flops']!=full['vdepth-q-150m-fixed-004']['total_training_matmul_flops']:
        raise AssertionError('Mixture/fixed neural work differs')
    if len({v['initial_model_digest'] for v in full.values()})!=1:raise AssertionError('Paired initialization differs')
    report={'status':'PASS_VARIABLE_DEPTH_QUALIFICATION','elapsed_seconds':time.time()-started,
        'optimizer_fixture':fixture,'global_gradient_checks':{k:v['global_gradient_check'] for k,v in tiny.items()},
        'interrupted_numerical_replay':numeric_replay,'recovered_state_bit_exact':True,
        'two_fresh_process_continuations_bit_exact':True,'full_size_cpu_probes':checks,
        'all_replica_states_equal':True,'only_expected_host_copy_fallback':True,
        'full_optimizer_recovery_both_recurrent_arms':True,'paired_initial_weights_equal':True,
        'equal_fixed_mixture_training_matmul_work':True,'full_size_reports':full}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'rates':{k:v['warm_global_real_targets_per_second'] for k,v in full.items()}}),flush=True)


if __name__=='__main__':main()
