"""Queued, bounded topology benchmark; never interrupts the scientific queue."""
import argparse
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def prerequisite_state(primary, annex):
    if primary is None:return 'WAIT'
    if primary['status']!='COMPLETE_VARIABLE_DEPTH_CAMPAIGN':return 'STOP_PRIMARY_FAILED'
    if annex is None:return 'WAIT'
    if annex['status'] not in ['COMPLETE_READONLY_DEPTH_DIAGNOSTICS','UNEXECUTED_NATIVE_TIME_GATE']:
        return 'STOP_ANNEX_INCOMPLETE'
    return 'READY'


def paired_warm_rate(reports):
    if len(reports)!=2 or any(r['warm_updates']!=100 for r in reports):
        raise ValueError('Require two completed 100-warm-update jobs')
    if any(not all(math.isfinite(r[k]) for k in ['warm_started_unix','warm_ended_unix','warm_real_targets'])
           or r['warm_real_targets']<=0 for r in reports):raise ValueError('Nonfinite/empty timing')
    begin=min(r['warm_started_unix'] for r in reports)
    end=max(r['warm_ended_unix'] for r in reports)
    if end<=begin or any(r['warm_ended_unix']<=r['warm_started_unix'] for r in reports):
        raise ValueError('Invalid concurrent timing interval')
    return {'real_targets':sum(r['warm_real_targets'] for r in reports),
            'wall_seconds':end-begin,'real_targets_per_second':sum(r['warm_real_targets'] for r in reports)/(end-begin),
            'scope':'Actual pair completion interval; individual rates are not summed.'}


def live_native_workers():
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=path.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'scripts.train_variable_depth' in parts or b'scripts.evaluate_depth_curve' in parts:
            found.append(int(path.parent.name))
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--annex',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comparison-arm',choices=['mixture','fixed'],default='mixture')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    arms=['ordinary',a.comparison_arm]
    prefix='fixedtop-q-v1' if a.comparison_arm=='fixed' else 'depthtop-q-v1'
    croot='/work/instance-nvme'
    public=Path('results/fixed_topology_qualification_20261024' if a.comparison_arm=='fixed'
                else 'results/depth_topology_qualification_20261024');public.mkdir(exist_ok=True)
    sources=json.loads((a.campaign/'plan.json').read_text())['source_sha256'].copy()
    for f in ['scripts/qualify_depth_topology.py','scripts/supervise.py','scripts/run_native.sh',
              'scripts/qualify_native.py','compute_trainium/depth_topology.py']:
        sources[f]=file_sha256(f)
    wait_deadline=dt.datetime(2026,10,24,4,25,tzinfo=dt.timezone.utc).timestamp()
    plan={'identity':'fixed-topology-q-v1' if a.comparison_arm=='fixed' else 'depth-topology-q-v1',
          'comparison_arm':a.comparison_arm,'purpose':'SOFTWARE_QUALIFICATION_ONLY',
          'declared_unix':time.time(),'wait_deadline_unix':wait_deadline,'execution_budget_seconds':2100,
          'maximum_native_attempts':8,'source_sha256':sources,'physical_chips_per_job':8,
          'layouts':[{'world':32,'batch':4},{'world':16,'batch':8}],'accumulation':2,'context':1024,
          'precision':'unscaled_bf16_with_fp32_weights_and_moments','optimizer_recipe':'pretraining',
          'cpu_loss_absolute_tolerance':.01,'cpu_gradient_relative_l2_tolerance':.01,
          'promotion_minimum_pair_rate_ratio':1.10,'promotion_minimum_each_arm_rate_ratio':.95,
          'scope':'Same source windows, depth assignments, initialization and estimated NN work. Fresh random qualification models only. No scientific weights or schedules changed. Native globals and full-size probes, exact stored-state recovery and replicas gate timing. Cross-layout multi-step trajectory identity is not claimed.'}
    write_state(a.output/'plan.json',plan);write_state(public/'plan.json',plan)
    launched=[];jobs=[];execution_started=None;cpu_checks=[]
    def costs():
        closed=[];missing=[]
        for name in launched:
            path=a.root/'snapshots'/name/'closure.json'
            if not path.exists():missing.append(name);continue
            row=json.loads(path.read_text())
            closed.append({'identity':name,'wall_seconds':row['wall_seconds'],
                'assigned_chip_hours':row['assigned_chip_hours'],'exit_code':row['exit_code']})
        return {'closed_attempts':closed,'missing_closures':missing,
                'assigned_chip_hours':sum(r['assigned_chip_hours'] for r in closed),
                'execution_opportunity_chip_hours':0 if execution_started is None else 16*(time.time()-execution_started)/3600,
                'scope':'All qualification attempts, including setup, CPU probes, compilation, updates and recovery. Prerequisite waits overlap the prior campaign and are not charged again here. Not the entire instance bill.'}
    def frozen():
        for f,sha in sources.items():
            if file_sha256(f)!=sha:raise AssertionError('Qualification source changed: '+f)
    def phase(name,world,batch,policy,half,steps,*,tiny=False,check_global=False,resume=None,peers=None):
        frozen()
        if len(launched)>=8 or time.time()+120>deadline:raise TimeoutError('Bounded qualification budget')
        cores=[half*32+c for c in range(32) if world==32 or c%4<2]
        if len(cores)!=world:raise AssertionError('Logical-core assignment differs')
        args=['--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
              '--context','1024','--batch',str(batch),'--accumulation','2','--architecture','preln',
              '--precision','bf16','--data-seed','23','--seed','103','--optimizer','fused','--recipe','pretraining',
              '--reference-threads','4','--warmup','4','--save','--purpose','qualification',
              '--size','tiny' if tiny else 'small','--depth-policy',policy,'--steps',str(steps)]
        if check_global:args+=['--check-global']
        if resume:args+=['--resume',f'{croot}/runs/{resume}/checkpoint.pt']
        if peers:args+=['--warm-rendezvous',f'{croot}/{a.output.relative_to(a.root)}/warm-{world}',
                        '--warm-peer-names',','.join(peers)]
        env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT=croot,RESEARCH_RUN=name,
            RESEARCH_ENTRY_MODULE='scripts.train_variable_depth',RESEARCH_DDP_RANKS=str(world),
            NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),RESEARCH_TIMEOUT_SECONDS=str(min(900,int(deadline-time.time()))))
        log=(a.output/(name+'.log')).open('w')
        proc=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();launched.append(name);jobs.append(proc)
        write_state(a.output/'state.json',{'status':'RUNNING','phase':name,'launched':launched,'deadline_unix':deadline})
        return name,proc
    def finish(items):
        reports={}
        for name,proc in items:
            code=proc.wait(timeout=max(1,deadline-time.time()))
            if code:raise RuntimeError('Native qualification failed: '+name)
            report=json.loads((a.root/'runs'/name/'report.json').read_text())
            if report['status']!='PASS_PACKED_TRAINING_PHASE' or not report['all_replica_model_optimizer_states_equal']:
                raise AssertionError('Incomplete update or replica qualification')
            probes=list((a.root/'runs'/name).glob('cpu-probe-rank*-depth*.json'))
            if len(probes)!=len(set(report['rank_depths'])):raise AssertionError('Missing depth CPU probes')
            for f in probes:
                check=json.loads(f.read_text())
                if not all(math.isfinite(check[k]) for k in ['native_loss','cpu_loss','gradient_relative_l2']) or abs(check['native_loss']-check['cpu_loss'])>.01 or check['gradient_relative_l2']>.01:
                    raise AssertionError('CPU numerical qualification failed: '+name)
                cpu_checks.append({'identity':name,'probe':f.name,**check})
            reports[name]=report
        return reports
    try:
        while True:
            def result(path):return json.loads(path.read_text()) if path.exists() else None
            state=prerequisite_state(result(a.campaign/'report.json'),result(a.annex/'report.json'))
            if state.startswith('STOP_'):
                report={'status':'UNEXECUTED_'+state,'launched':[],'execution_costs':costs()}
                write_state(a.output/'report.json',report);write_state(public/'report.json',report);return
            monitor=result(a.campaign/'liveness.json')
            if state=='READY' and monitor and monitor['status']=='CAMPAIGN_CLOSED' and not live_native_workers():break
            if time.time()>wait_deadline:
                report={'status':'UNEXECUTED_WAIT_DEADLINE','launched':[],'execution_costs':costs()}
                write_state(a.output/'report.json',report);write_state(public/'report.json',report);return
            write_state(a.output/'state.json',{'status':'WAITING_FOR_SCIENCE_AND_ANNEX','checked_unix':time.time()})
            time.sleep(10)
        frozen();execution_started=time.time();deadline=execution_started+2100
        write_state(a.output/'execution-start.json',{'started_unix':execution_started,'deadline_unix':deadline})
        tiny=finish([phase(prefix+'-tiny-w32',32,4,a.comparison_arm,0,8,tiny=True,check_global=True),
                     phase(prefix+'-tiny-w16',16,8,a.comparison_arm,1,8,tiny=True,check_global=True)])
        if any(not math.isfinite(r['global_gradient_check']['gradient_relative_l2']) or r['global_gradient_check']['gradient_relative_l2']>.01 for r in tiny.values()):
            raise AssertionError('Global objective agreement failed')
        full={};pairs={}
        for world,batch in [(32,4),(16,8)]:
            peers=[f'{prefix}-full-{arm}-w{world}' for arm in arms]
            found=finish([phase(name,world,batch,arm,half,104,peers=peers)
                          for half,(name,arm) in enumerate(zip(peers,arms))])
            if any(r['parameters']!=150436608 or r['warm_updates']!=100 for r in found.values()):
                raise AssertionError('Full-size warm benchmark differs')
            full.update(found);pairs[str(world)]=paired_warm_rate(list(found.values()))
        if len({r['initial_model_digest'] for r in full.values()})!=1:
            raise AssertionError('Benchmark initial weights differ')
        restored=finish([phase(f'{prefix}-restore-{arm}-w16',16,8,arm,half,8,
                            resume=f'{prefix}-full-{arm}-w16')
                         for half,arm in enumerate(arms)])
        for arm in arms:
            old=full[f'{prefix}-full-{arm}-w16'];new=restored[f'{prefix}-restore-{arm}-w16']
            if new['recovered_model_optimizer_digest']!=old['state_digest']:
                raise AssertionError('Full optimizer recovery differs')
        for arm in arms:
            x,y=[full[f'{prefix}-full-{arm}-w{w}'] for w in [32,16]]
            if (x['phase_real_targets'],x['total_training_matmul_flops'])!=(y['phase_real_targets'],y['total_training_matmul_flops']):
                raise AssertionError('Resized source or compute budget differs')
        ratio=pairs['16']['real_targets_per_second']/pairs['32']['real_targets_per_second']
        arm_ratios={arm:full[f'{prefix}-full-{arm}-w16']['warm_global_real_targets_per_second']/
                         full[f'{prefix}-full-{arm}-w32']['warm_global_real_targets_per_second'] for arm in arms}
        report={'status':'PASS_DEPTH_TOPOLOGY_QUALIFICATION','execution_wall_seconds':time.time()-execution_started,
                'source_plan_sha256':file_sha256(a.output/'plan.json'),'tiny_global_gradient_checks':{k:r['global_gradient_check'] for k,r in tiny.items()},
                'cpu_probes':cpu_checks,'execution_costs':costs(),
                'full_size_reports':full,'full_optimizer_recovery':True,'all_replica_states_equal':True,
                'paired_warm_completion':pairs,'pair_rate_ratio_16_over_32':ratio,'per_arm_rate_ratios':arm_ratios,
                'promote_for_future_use':ratio>=1.10 and min(arm_ratios.values())>=.95,'launched':launched,
                'scope':plan['scope']+' Warm timing excludes numerical setup, compilation and checkpointing; all attempts remain in launch costs. No scientific performance claim or automatic model training.'}
        write_state(a.output/'report.json',report);write_state(public/'report.json',report)
    except Exception as error:
        # Each native supervisor retains its fail-closed resource/compile guards.
        # Terminate only the dedicated container if a timed-out native job remains.
        if any(proc.poll() is None for proc in jobs):
            subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
        report={'status':'STOPPED_DEPTH_TOPOLOGY_QUALIFICATION','error':str(error),'launched':launched,
                'execution_wall_seconds':None if execution_started is None else time.time()-execution_started,
                'execution_costs':costs()}
        write_state(a.output/'report.json',report);write_state(public/'report.json',report)
        raise


if __name__=='__main__':main()
