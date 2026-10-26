"""Finish only the original One endpoints under explicit exact-state recovery."""
import argparse
import concurrent.futures as cf
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_gradient_point import point_tables
from compute_trainium.future_span_recovery import rebind_stable_checkpoint
from compute_trainium.future_span_scientific_state import signature, validate_checkpoint
from compute_trainium.metrics import DocumentScore, paired_group_interval
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint, file_sha256
from scripts.qualify_native import digest
from scripts.run_fixed_control_replication import reserves

IDENTITY='one-control-recovery-v1'
PUBLIC=Path('results/one_control_recovery_20261026')
DOC='docs/ONE_CONTROL_RECOVERY_20261026.md'
KEYS=['one-s17293-data','one-s17293-work']


def load(path):return json.loads(Path(path).read_text())


def register(root):
    out=root/IDENTITY
    if (out/'plan.json').exists():return load(out/'plan.json')
    if out.exists():raise FileExistsError('Preserve partial preparation before explicit recovery')
    started=time.time();reserves(root,admission=True);torch.set_num_threads(12)
    old_dir=root/'fixed-control-replication-main-v1';old_file=old_dir/'full-plan.json';old=load(old_file)
    stopped=load(old_dir/'report.json')
    if stopped['status']!='STOPPED_FIXED_CONTROL_DEVELOPMENT_REPLICATION':raise ValueError('Preserved stopped predecessor required')
    prefix_dir=root/'runs/fixed-control-replication-main-v1-one-s17293-stable'
    prefix_report=load(prefix_dir/'report.json');record=next(v for v in prefix_report['checkpoint_records'] if v['update']==1920)
    prefix=prefix_dir/'checkpoint-u1920.pt'
    if (prefix_report['update']!=1920 or not record['all_complete_replica_states_equal']
            or file_sha256(prefix)!=record['checkpoint_sha256']):raise AssertionError('Exact completed stable prefix required')
    old_sha=file_sha256(old_file);old_case=old['cases'][KEYS[0]]
    counts,anchors,depths,work=point_tables(root,old_file,old,old_case)
    if not np.all(depths==1):raise AssertionError('One-pass source/depth path must be unchanged')
    saved=torch.load(prefix,map_location='cpu',mmap=True,weights_only=False)
    domain_counts=list(map(int,counts[saved['cursor']]-counts[0]))
    validate_checkpoint(saved,KEYS[0],old_case,signature(old_sha,old,old_case),
        expected_targets=sum(domain_counts),expected_domains=domain_counts,
        expected_matrix_ops=int(work[:1920].sum()),expected_anchors=int(anchors[saved['cursor']]))
    if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=record['full_state_digest']:
        raise AssertionError('Original complete model/optimizer state differs')
    extras=['scripts/run_one_control_recovery.py','compute_trainium/future_span_recovery.py',
            'tests/test_future_span_recovery.py',DOC]
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in extras}}
    if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Qualified source changed')
    cases={key:copy.deepcopy(old['cases'][key]) for key in KEYS}
    cases[KEYS[0]]['gradient_updates']=[1920,3840]
    phase_names=[IDENTITY+'-prefix-gradient']
    for key in KEYS:phase_names.extend([IDENTITY+'-'+key+'-finish',IDENTITY+'-'+key+'-gradient'])
    ports=reserve_ports(root,phase_names);worker,host=split_snapshot_sources(sources)
    child=copy.deepcopy(old);child.update(identity=IDENTITY+'-science',registered_unix=time.time(),
        cases=cases,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=ports,scope=Path(DOC).read_text())
    for entry in child['initial_states'].values():entry['file']=os.path.relpath(old_dir/entry['file'],out)
    for case in cases.values():
        for field in ['depth_table','work_table']:case[field]=os.path.relpath(old_dir/case[field],out)
    rate=load(old_dir/'plan.json')['measured_budget']['warm_seconds_per_update']['one']
    predicted=2705*rate*1.5+300+300+900+300+1260+300
    cutoff=load(old_dir/'execution/execution-start.json')['deadline_unix']
    if predicted>9000 or time.time()+predicted>cutoff:raise RuntimeError('Complete measured recovery does not fit original cutoff')
    out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=True);child_file=out/'science-plan.json';write_state(child_file,child)
    child_sha=file_sha256(child_file)
    rebound=rebind_stable_checkpoint(saved,signature(old_sha,old,old_case),signature(child_sha,child,cases[KEYS[0]]),old_case,update=1920)
    before=digest({k:v for k,v in saved.items() if k!='signature'})
    migrated=out/'stable-u1920-rebound.pt';atomic_checkpoint(migrated,rebound)
    check=torch.load(migrated,map_location='cpu',mmap=True,weights_only=False)
    if digest({k:v for k,v in check.items() if k!='signature'})!=before or check['signature']!=rebound['signature']:
        raise AssertionError('Recovery migration changed tensors,RNG,counters or other metadata')
    for key in KEYS:
        case=cases[key]
        validate_checkpoint(check,key,case,signature(child_sha,child,case),
            expected_targets=sum(domain_counts),expected_domains=domain_counts,
            expected_matrix_ops=int(work[:1920].sum()),expected_anchors=int(anchors[check['cursor']]))
    migration=dict(status='PASS_SIGNATURE_ONLY_STABLE_CHECKPOINT_MIGRATION',
        original_checkpoint=str(prefix.relative_to(root)),original_checkpoint_sha256=record['checkpoint_sha256'],
        original_plan_sha256=old_sha,new_plan_sha256=child_sha,new_checkpoint=migrated.name,
        new_checkpoint_sha256=file_sha256(migrated),all_other_state_exact=True,
        original_complete_state_digest=record['full_state_digest'],update=1920,optimizer_updates=0,
        scope='Only signature.plan_sha256 changed; original file retained. Model/optimizer,rank RNG,source counters,schedule and all other metadata exactly verified after disk readback. Future trajectory identity is not established.')
    write_state(out/'migration.json',migration);write_state(PUBLIC/'migration.json',migration)
    recurrence_docs=old_dir/'scores/fixed-s17293-data/document-scores.json'
    recurrence_gate=old_dir/'gradients/fixed-s17293-data/report.json'
    if load(recurrence_gate)['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT':raise ValueError('Pinned recurrent endpoint must pass')
    plan=dict(identity=IDENTITY,source_sha256=sources,root_comm_ports=ports,
        child_plan=dict(file=child_file.name,sha256=child_sha),cases=cases,
        prefix_checkpoint=dict(file=migrated.name,sha256=migration['new_checkpoint_sha256']),
        migration_sha256=file_sha256(out/'migration.json'),predecessor_report_sha256=file_sha256(old_dir/'report.json'),
        recurrence_documents_sha256=file_sha256(recurrence_docs),recurrence_gate_sha256=file_sha256(recurrence_gate),
        diagnosis_identity='mixing-precision-diagnosis-v1',seconds_limit=9000,phase_seconds_limit=5000,
        cpu_phase_seconds_limit=1200,absolute_deadline_unix=cutoff,
        measured_budget=dict(warm_seconds_per_update=rate,slowdown_factor=1.5,
            predicted_seconds_with_margin=predicted,cap_seconds=9000),
        actual_new_research_updates=4625,conceptual_histories_include_stable_prefix=True,
        cpu_preparation_seconds=time.time()-started,test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);write_state(PUBLIC/'science-plan.json',child)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_RECOVERY_NOT_ADMITTED',assigned_chips=0))
    return plan


class CheckedExecution(Execution):
    def check(self):
        super().check();reserves(self.root)
        if time.time()>self.plan['absolute_deadline_unix']:raise TimeoutError('Original main absolute cutoff retained')
        child=self.plan['child_plan']
        if file_sha256(self.plan_file.parent/child['file'])!=child['sha256']:raise AssertionError('Recovery scientific plan changed')


def run(root):
    out=root/IDENTITY;file=out/'plan.json';plan=load(file);child=plan['child_plan'];science=out/child['file']
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal recovery')
    diagnosis_file=root/plan['diagnosis_identity']/'report.json'
    if not diagnosis_file.exists() or load(diagnosis_file)['status']!='COMPLETE_MIXING_PRECISION_DIAGNOSIS':
        raise RuntimeError('Numerical diagnosis must be terminal before separate One recovery admission')
    if managed_running():raise RuntimeError('All predecessor owners must be closed')
    reserves(root,admission=True)
    if time.time()+plan['measured_budget']['predicted_seconds_with_margin']>plan['absolute_deadline_unix']:
        raise RuntimeError('Full remaining measured budget no longer fits original cutoff')
    predecessor=root/'fixed-control-replication-main-v1'
    for path,sha in [(predecessor/'report.json',plan['predecessor_report_sha256']),
                     (out/'migration.json',plan['migration_sha256']),
                     (out/plan['prefix_checkpoint']['file'],plan['prefix_checkpoint']['sha256']),
                     (predecessor/'scores/fixed-s17293-data/document-scores.json',plan['recurrence_documents_sha256']),
                     (predecessor/'gradients/fixed-s17293-data/report.json',plan['recurrence_gate_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Recovery prerequisite changed')
    write_state(out/'admission-prerequisite.json',dict(diagnosis_report_sha256=file_sha256(diagnosis_file)))
    execution=CheckedExecution(root,file,out/'execution');result=dict(status='STOPPED_ONE_CONTROL_RECOVERY')
    scores={};gates={};fits={};cpu=cf.ThreadPoolExecutor(3);diagnostics=cf.ThreadPoolExecutor(2)
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Owned recovery interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def gradient(key,checkpoint,sha,phase,suffix):
        common=['--plan-sha',child['sha256'],'--case',key,'--checkpoint-sha',sha]
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/'+science.name,
              *common,'--checkpoint','/work/instance-nvme/'+str(checkpoint.relative_to(root)),
              '--comm-port',str(plan['root_comm_ports'][phase])]
        native=execution.native(phase,'scripts.capture_future_span_primitives',opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        target=out/'gradients'/suffix
        opts=['--root',str(root),'--plan',str(science),*common,'--checkpoint',str(checkpoint),
              '--capture',str(root/'runs'/phase),'--output',str(target),
              '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        independent=execution.cpu(suffix+'-gradient','scripts.audit_future_span_gradient',opts,target/'report.json',
                                  'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        return dict(native_report_sha256=file_sha256(root/'runs'/phase/'report.json'),independent_cpu=independent)
    def fit(key):
        name=IDENTITY+'-'+key+'-finish';case=plan['cases'][key];prefix=plan['prefix_checkpoint']
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/'+science.name,
              '--plan-sha',child['sha256'],'--case',key,'--segment-end',str(case['total_updates']),
              '--comm-port',str(plan['root_comm_ports'][name]),'--resume',
              '/work/instance-nvme/'+IDENTITY+'/'+prefix['file'],'--resume-sha',prefix['sha256']]
        value=execution.native(name,'scripts.train_future_span_primitives',opts,'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        point=next(v for v in value['checkpoint_records'] if v['update']==case['total_updates'])
        path=root/'runs'/name/f'checkpoint-u{case["total_updates"]}.pt'
        if (value['plan_sha256']!=child['sha256'] or value['case_id']!=key or value['update']!=case['total_updates']
                or not value['complete_state_on_load_exact'] or not value['cpu_rng_on_load_exact']
                or value['recovered_state_digest']!=load(out/'migration.json')['original_complete_state_digest']
                or not point['all_complete_replica_states_equal'] or file_sha256(path)!=point['checkpoint_sha256']
                or value['real_targets']!=case['end_targets'] or value['valid_anchors']!=case['end_anchors']
                or value['estimated_dense_matrix_ops']!=case['training_dense_matrix_ops']
                or value['research_optimizer_updates']!=case['total_updates']-1920):
            raise AssertionError('Recovered endpoint identity,state or conceptual work differs')
        return name,value,path,point['checkpoint_sha256']
    def score(key,capture):
        _,value,path,sha=capture;target=out/'scores'/key
        opts=['--root',str(root),'--plan',str(science),'--plan-sha',child['sha256'],'--case',key,
              '--checkpoint',str(path),'--checkpoint-sha',sha,'--cohort','curve','--depth','1',
              '--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        return execution.cpu(key+'-score','scripts.evaluate_future_span_cpu',opts,target/'report.json',
                             'COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
    def endpoint(key,capture):
        future=cpu.submit(score,key,capture);future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        gate=gradient(key,capture[2],capture[3],IDENTITY+'-'+key+'-gradient',key)
        return future.result(),gate
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_STABLE_PREFIX_GRADIENT_GATE',maximum_chips=8))
        gates['stable-prefix']=gradient(KEYS[0],out/plan['prefix_checkpoint']['file'],plan['prefix_checkpoint']['sha256'],
                                       IDENTITY+'-prefix-gradient','stable-prefix')
        write_state(PUBLIC/'status.json',dict(status='RUNNING_ORIGINAL_ONE_CONTINUATIONS',maximum_chips=16))
        handles={}
        with cf.ThreadPoolExecutor(2) as pool:
            futures={pool.submit(fit,key):key for key in KEYS}
            for future in futures:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(futures):
                key=futures[future];fits[key]=future.result();handles[key]=diagnostics.submit(endpoint,key,fits[key])
                handles[key].add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        for key,future in handles.items():scores[key],gates[key]=future.result()
        if sum(v[1]['research_optimizer_updates'] for v in fits.values())!=4625:raise AssertionError('New physical update count differs')
        def documents(path):return [DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets']) for v in load(path)['records']]
        reference=documents(predecessor/'scores/fixed-s17293-data/document-scores.json')
        comparisons={key:paired_group_interval(documents(out/'scores'/key/'document-scores.json'),reference,seed=17293,
                                              weights=load(science)['evaluation']['weights']) for key in KEYS}
        result=dict(status='COMPLETE_ORIGINAL_ONE_CONTROL_RECOVERY',scores=scores,gradient_gates=gates,
            comparisons= comparisons,actual_new_research_updates=4625,
            training_records={k:dict(identity=v[0],report_sha256=file_sha256(root/'runs'/v[0]/'report.json'),
                                   checkpoint_sha256=v[3]) for k,v in fits.items()},
            cheap_control_repaired=False,automatic_followup_admitted=False,scope=plan['scope'])
        execution.check()
    except BaseException as error:execution.failed.set();result.update(error=str(error),scores=scores,gradient_gates=gates);raise
    finally:
        execution.failed.set();diagnostics.shutdown(wait=True,cancel_futures=True);cpu.shutdown(wait=True,cancel_futures=True)
        execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    register(a.root)
    if not a.prepare_only:run(a.root)


if __name__=='__main__':main()
