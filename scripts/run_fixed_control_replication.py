"""Finite stronger-control comparison and one new development seed."""
import argparse
import concurrent.futures as cf
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import validate_curve, stop_on_failed_future
from compute_trainium.future_span_execution import Execution
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.metrics import DocumentScore, paired_group_interval
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY = 'fixed-control-replication-main-v1'
PREP = 'fixed-control-replication-preparation-v1'
PUBLIC = Path('results/fixed_control_replication_20261026/main')
DOC = 'docs/FIXED_CONTROL_REPLICATION_ADMISSION_20261026.md'
COMPLETE = 'COMPLETE_FIXED_CONTROL_DEVELOPMENT_REPLICATION'


def reserves(root, *, admission=False):
    require_inode_reserve([root, Path('/')])
    if shutil.disk_usage(root).free < (70 if admission else 35)*1024**3:
        raise RuntimeError('Protected NVMe storage reserve failed')
    if shutil.disk_usage('/').free < 35*1024**3:
        raise RuntimeError('Protected root storage reserve failed')
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:'))
    if available < 512*1024**3:
        raise RuntimeError('Protected available-memory reserve failed')


def load(path):
    return json.loads(Path(path).read_text())


def register(root):
    out = root/IDENTITY
    if (out/'plan.json').exists():
        return load(out/'plan.json')
    reserves(root, admission=True)
    if managed_running():
        raise RuntimeError('Predecessor native owners must be terminal')
    parent_file = root/'fixed-two-ntp-main-v1/plan.json'
    parent, finished = load(parent_file), load(parent_file.parent/'report.json')
    prep_file = root/PREP/'manifest.json'; prep = load(prep_file)
    tune_file = root/'fixed-two-mixing-tune-v1/report.json'; tune = load(tune_file)
    tuning_plan = load(tune_file.parent/'plan.json')
    if (finished['status'] != 'COMPLETE_FIXED_TWO_NTP_CURVE'
            or finished['plan_sha256'] != file_sha256(parent_file)
            or prep['status'] != 'PASS_FIXED_CONTROL_REPLICATION_CPU_PREPARATION_ONLY'
            or prep['parent_plan_sha256'] != file_sha256(parent_file)
            or prep['parent_report_sha256'] != file_sha256(parent_file.parent/'report.json')):
        raise ValueError('Complete immutable parent and CPU preparation required')
    if (tune['status'] != 'COMPLETE_FIXED_TWO_MIXING_CONTROL_TUNING'
            or tune['plan_sha256'] != file_sha256(tune_file.parent/'plan.json')
            or set(tune['scores']) != set(tuning_plan['cases'])
            or len(tune['prefix_gates']) != 3):
        raise ValueError('Complete frozen cheap-control selection required')
    winner = min(tune['scores'], key=lambda k: (tune['scores'][k]['metrics']['weighted_nll'],
                                              tuning_plan['cases'][k]['peak_lr']))
    if (winner != tune['selected_case'] or tune['selected_rate'] != tuning_plan['cases'][winner]['peak_lr']
            or tune['selected_final_gate']['independent_cpu']['case_id'] != winner
            or any(g['independent_cpu']['status'] != 'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                   for g in [*tune['prefix_gates'].values(), tune['selected_final_gate']])):
        raise AssertionError('Complete-grid winner or full-gradient gates differ')
    sources = {}
    for group in [parent['source_sha256'], tuning_plan['source_sha256'], prep['source_sha256'],
                  {f: file_sha256(f) for f in ['scripts/run_fixed_control_replication.py', DOC]}]:
        for path, sha in group.items():
            if path in sources and sources[path] != sha:
                raise AssertionError('Qualified frozen source versions conflict: '+path)
            if file_sha256(path) != sha:
                raise AssertionError('Qualified source changed: '+path)
            sources[path] = sha
    if any(prep[k] != parent[k] for k in ['model', 'future_config', 'data', 'evaluation']):
        raise AssertionError('Prepared family or source differs')
    from compute_trainium.future_span import FutureSpanConfig
    from compute_trainium.mixing_control_science import matrix_work, VARIANT
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    initial = copy.deepcopy(prep['initial_states'])
    for entry in initial.values():
        file = root/entry['file']
        if file_sha256(file) != entry['sha256']:
            raise AssertionError('Original random initializer changed')
        entry['file'] = os.path.relpath(file, out)
    counts_file = root/parent['data']['packed']/'prefix_counts.npy'
    metadata = load(counts_file.parent/'manifest.json')
    if (file_sha256(counts_file.parent/'manifest.json') != parent['data']['manifest_sha256']
            or file_sha256(counts_file) != metadata['files']['prefix_counts.npy']):
        raise AssertionError('Original source counts changed')
    counts = np.load(counts_file, mmap_mode='r')
    aux = parent['auxiliary_data']; anchors_file = root/aux['directory']/aux['prefix_file']
    if file_sha256(anchors_file) != aux['prefix_sha256']:
        raise AssertionError('Original auxiliary-label counters changed')
    anchors = np.load(anchors_file, mmap_mode='r')
    mixing_work = root/PREP/'mixing-work.npy'
    if file_sha256(mixing_work) != prep['mixing_work_sha256']:
        raise AssertionError('Prepared cheap work table changed')
    original = load(root/'future-span-study-preparation-v1/manifest.json')
    depths = np.load(parent_file.parent/parent['cases']['fixed-2b']['depth_table'], mmap_mode='r')
    recomputed = matrix_work(ParcaeCandidateConfig(**parent['model']), FutureSpanConfig(**parent['future_config']),
                             depths, world=32, optimizer_per_replica=original['optimizer_matrix_ops_per_replica'])
    if not np.array_equal(recomputed, np.load(mixing_work, mmap_mode='r')) or not np.all(depths == 2):
        raise AssertionError('Independently recomputed cheap work or fixed depth differs')
    cases, variants, work_matches = {}, {}, {}
    for entry in prep['cases']:
        key, variant = entry['case_id'], entry['variant']
        template = 'fixed-2b' if variant != 'one' else ('one-work-2b' if key.endswith('-work') else 'one-2b')
        case = copy.deepcopy(parent['cases'][template]); case.pop('parent', None)
        total = entry['total_updates']; seed = entry['seed']
        case.update(seed=seed, state_seed=seed+2009, initial_state=f's{seed}-family',
                    peak_lr=tune['selected_rate'] if variant == 'mixing' else entry['peak_lr'],
                    total_updates=total, end_targets=entry['real_targets'],
                    end_anchors=int(anchors[total*512]-anchors[0]),
                    cooldown_start_update=total//2, cooldown_start_targets=entry['cooldown_start_targets'],
                    checkpoint_updates=[total], segment_ends=[total], gradient_updates=[total])
        for field in ['depth_table', 'work_table']:
            file = mixing_work if variant == 'mixing' and field == 'work_table' else parent_file.parent/case[field]
            if variant != 'mixing' or field != 'work_table':
                if file_sha256(file) != case[field+'_sha256']:
                    raise AssertionError('Original fixed table changed')
            case[field] = os.path.relpath(file, out); case[field+'_sha256'] = file_sha256(file)
        # Admission validates inputs before creating its output directory.
        work = np.load((out/case['work_table']).resolve(), mmap_mode='r')
        case['training_dense_matrix_ops'] = int(work[:total].sum())
        if case['end_targets'] != int((counts[total*512]-counts[0]).sum()):
            raise AssertionError('Prepared exact target endpoint changed')
        cases[key], variants[key] = case, variant
        if key.endswith('-work'):
            budget = prep['reference_target_matrix_work']; actual = case['training_dense_matrix_ops']
            if not actual <= budget < actual+int(work[total]):
                raise AssertionError('Exact integer work floor required')
            work_matches[key] = dict(target=budget, actual=actual, relative_shortfall=(budget-actual)/budget)
    if len(cases) != 5 or prep['fresh_development_seed'] != 17293:
        raise AssertionError('Preserve exactly the five prepared endpoints')
    cases['one-s17293-data'].update(segment_ends=[1920,3840], checkpoint_updates=[1920,3840])
    cases['one-s17293-work']['parent'] = dict(case_id='one-s17293-data', update=1920)
    nodes = {key+'-finish': dict(case=key,end=case['total_updates'],parent=None) for key,case in cases.items()}
    nodes['one-s17293-stable'] = dict(case='one-s17293-data',end=1920,parent=None)
    for key in ['one-s17293-data','one-s17293-work']:
        nodes[key+'-finish']['parent'] = 'one-s17293-stable'
    actual_updates = validate_curve(cases,nodes)
    if actual_updates != prep['proposed_actual_updates'] or actual_updates != 19573:
        raise AssertionError('Actual shared-prefix work changed')
    if any(w['warm_updates'] != 100 for w in tune['warm'].values()):
        raise AssertionError('Actual 100-update cheap measurements required')
    rates = dict(one=parent['measured_budget']['one_warm_seconds_per_update'],
                 fixed=parent['measured_budget']['fixed_warm_seconds_per_update'],
                 mixing=max(w['warm_seconds']/100 for w in tune['warm'].values()))
    if not all(math.isfinite(v) and v>0 for v in rates.values()):
        raise AssertionError('Complete positive measured timings required')
    duration = sum((n['end']-(nodes[n['parent']]['end'] if n['parent'] else 0))*rates[variants[n['case']]]
                   for n in nodes.values())
    cold = max(300., parent['measured_budget']['cold_allowance'],tuning_plan['measured_budget']['cold_allowance'])
    estimate = duration*1.5/2+(len(nodes)+len(cases))*cold/2+5*900/3+5*360/3+300
    if estimate > 21600:
        raise RuntimeError('Complete stage does not fit its measured six-hour cap')
    phases = [IDENTITY+'-'+key for key in nodes]+[IDENTITY+'-'+key+'-gradient' for key in cases]
    ports = reserve_ports(root, phases); worker, host = split_snapshot_sources(sources)
    out.mkdir(); PUBLIC.mkdir(parents=True)
    children = {}
    for family in ['full','mixing']:
        child = copy.deepcopy(parent)
        for stale in ['nodes','stratification_reference','work_matches','primary_comparison',
                      'primary_scope','planned_actual_research_updates','secondary_diagnostic_scope']:
            child.pop(stale,None)
        child.update(identity=IDENTITY+'-'+family,registered_unix=time.time(),stage='fixed_control_development',
                     source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
                     initial_states=initial,root_comm_ports=ports,
                     cases={k:v for k,v in cases.items() if (variants[k]=='mixing') == (family=='mixing')},
                     seconds_limit=21600,phase_seconds_limit=9000,cpu_phase_seconds_limit=1200,
                     scope=Path(DOC).read_text())
        if family=='mixing': child['model_variant']=VARIANT
        else: child.pop('model_variant',None)
        file = out/(family+'-plan.json'); write_state(file,child)
        children[family]=dict(file=file.name,sha256=file_sha256(file))
        write_state(PUBLIC/file.name,child)
    parent_scores = {k: file_sha256(parent_file.parent/'scores'/k/'document-scores.json')
                     for k in ['one-2b','one-work-2b','fixed-2b']}
    plan = dict(identity=IDENTITY,registered_unix=time.time(),source_sha256=sources,
                cases=cases,variants=variants,nodes=nodes,child_plans=children,root_comm_ports=ports,
                initial_states=initial,seconds_limit=21600,phase_seconds_limit=9000,cpu_phase_seconds_limit=1200,
                measured_budget=dict(warm_seconds_per_update=rates,cold_allowance=cold,
                    predicted_seconds_with_margin=estimate,cap_seconds=21600),
                planned_actual_research_updates=actual_updates,work_matches=work_matches,
                evaluation=parent['evaluation'],preparation_sha256=file_sha256(prep_file),
                tuning_report_sha256=file_sha256(tune_file),parent_report_sha256=file_sha256(parent_file.parent/'report.json'),
                parent_document_scores_sha256=parent_scores,
                test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=Path(DOC).read_text())
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_NOT_STARTED',assigned_chips=0))
    return plan


class CheckedExecution(Execution):
    def check(self):
        super().check(); reserves(self.root)
        for child in self.plan['child_plans'].values():
            if file_sha256(self.plan_file.parent/child['file']) != child['sha256']:
                raise AssertionError('Frozen scientific child plan changed')
        for path, sha in [(self.root/PREP/'manifest.json',self.plan['preparation_sha256']),
                          (self.root/'fixed-two-mixing-tune-v1/report.json',self.plan['tuning_report_sha256']),
                          (self.root/'fixed-two-ntp-main-v1/report.json',self.plan['parent_report_sha256'])]:
            if file_sha256(path) != sha:
                raise AssertionError('Frozen admission prerequisite changed')


def point(capture):
    return next(r for r in capture[1]['checkpoint_records'] if r['update']==capture[1]['update'])


def run(root, *, resume=False):
    out=root/IDENTITY;plan_file=out/'plan.json';plan=load(plan_file)
    if (out/'report.json').exists():
        result=load(out/'report.json')
        if result['status']==COMPLETE:return result
        raise FileExistsError('Preserve stopped execution; explicit recovery registration required')
    validate_curve(plan['cases'],plan['nodes'])
    execution=CheckedExecution(root,plan_file,out/'execution',resume=resume)
    cpu=cf.ThreadPoolExecutor(3);diagnostics=cf.ThreadPoolExecutor(2)
    captures={};scores={};gates={};handles={};records=[]
    result=dict(status='STOPPED_FIXED_CONTROL_DEVELOPMENT_REPLICATION')
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Owned control stage interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}

    def identity(key):
        mixing=plan['variants'][key]=='mixing'; child=plan['child_plans']['mixing' if mixing else 'full']
        return mixing,child,out/child['file']

    def fit(node_id,parent):
        node=plan['nodes'][node_id];key=node['case'];case=plan['cases'][key];name=IDENTITY+'-'+node_id
        mixing,child,file=identity(key)
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/'+file.name,
              '--plan-sha',child['sha256'],'--case',key,'--segment-end',str(node['end']),
              '--comm-port',str(plan['root_comm_ports'][name])]
        source=None
        if parent:
            source=point(parent);opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{source["update"]}.pt',
                                       '--resume-sha',source['checkpoint_sha256']]
        module='scripts.train_mixing_control_scientific_v2' if mixing else 'scripts.train_future_span_primitives'
        row=execution.native(name,module,opts,'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        if (row['plan_sha256']!=child['sha256'] or row['case_id']!=key or row['update']!=node['end']
                or row['qualification_only'] or row['objective_weight']!=0.):
            raise AssertionError('Scientific segment identity differs')
        for state in row['checkpoint_records']:
            if (not state['all_complete_replica_states_equal']
                    or file_sha256(root/'runs'/name/f'checkpoint-u{state["update"]}.pt')!=state['checkpoint_sha256']):
                raise AssertionError('Complete saved replica/checkpoint states differ')
        if parent and (not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact']
                       or row['recovered_state_digest']!=source['full_state_digest']):
            raise AssertionError('Exact full-state restore failed')
        if node['end']==case['total_updates'] and (row['real_targets'],row['valid_anchors'],row['estimated_dense_matrix_ops'])!=(case['end_targets'],case['end_anchors'],case['training_dense_matrix_ops']):
            raise AssertionError('Cooled endpoint accounting differs')
        return name,row

    def score(key,capture):
        name,row=capture;state=point(capture);mixing,child,file=identity(key);target=out/'scores'/key
        opts=['--root',str(root),'--plan',str(file),'--plan-sha',child['sha256'],'--case',key,
              '--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),
              '--checkpoint-sha',state['checkpoint_sha256'],'--cohort','curve',
              '--depth',str(plan['cases'][key]['primary_inference_depth']),'--output',str(target),
              '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        module='scripts.evaluate_mixing_prediction_v2' if mixing else 'scripts.evaluate_future_span_cpu'
        return execution.cpu(key+'-score',module,opts,target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')

    def gradient(key,capture):
        name,row=capture;state=point(capture);mixing,child,file=identity(key);phase=IDENTITY+'-'+key+'-gradient'
        common=['--plan-sha',child['sha256'],'--case',key,'--checkpoint-sha',state['checkpoint_sha256']]
        checkpoint=root/'runs'/name/f'checkpoint-u{row["update"]}.pt'
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/'+file.name,*common,
              '--checkpoint','/work/instance-nvme/'+str(checkpoint.relative_to(root)),
              '--comm-port',str(plan['root_comm_ports'][phase])]
        module='scripts.capture_mixing_control_gradient' if mixing else 'scripts.capture_future_span_primitives'
        native=execution.native(phase,module,opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        target=out/'gradients'/key
        opts=['--root',str(root),'--plan',str(file),*common,'--checkpoint',str(checkpoint),
              '--capture',str(root/'runs'/phase),'--output',str(target),
              '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        module='scripts.audit_mixing_scientific_gradient' if mixing else 'scripts.audit_future_span_gradient'
        gate=execution.cpu(key+'-gradient',module,opts,target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        # Raw global-batch capture rows stay private; publish the independent gate.
        return dict(native_status=native['status'],native_report_sha256=file_sha256(root/'runs'/phase/'report.json'),
                    independent_cpu=gate)

    def read_endpoint(key,capture):
        future=cpu.submit(score,key,capture);future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        gate=gradient(key,capture)
        return future.result(),gate

    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_FIXED_CONTROL_DEVELOPMENT_REPLICATION',maximum_chips=16))
        pending=set(plan['nodes']);running={}
        def priority(key):
            n=plan['nodes'][key];origin=plan['nodes'][n['parent']]['end'] if n['parent'] else 0
            return key.endswith('-stable'),(n['end']-origin)*plan['measured_budget']['warm_seconds_per_update'][plan['variants'][n['case']]],key
        with cf.ThreadPoolExecutor(2) as workers:
            while pending or running:
                execution.check()
                ready=sorted([k for k in pending if plan['nodes'][k]['parent'] is None or plan['nodes'][k]['parent'] in captures],key=priority,reverse=True)
                for key in ready[:2-len(running)]:
                    parent=plan['nodes'][key]['parent'];future=workers.submit(fit,key,captures[parent] if parent else None)
                    future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed));running[future]=key;pending.remove(key)
                if not running:raise RuntimeError('Frozen DAG cannot make progress')
                done,_=cf.wait(running,timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    node=running.pop(future);capture=future.result();captures[node]=capture
                    records.append(dict(node=node,identity=capture[0],report_sha256=file_sha256(root/'runs'/capture[0]/'report.json'),
                                        update=capture[1]['update'],research_optimizer_updates=capture[1]['research_optimizer_updates']))
                    write_state(PUBLIC/'training-progress.json',dict(records=records))
                    key=plan['nodes'][node]['case']
                    if plan['nodes'][node]['end']==plan['cases'][key]['total_updates']:
                        handles[key]=diagnostics.submit(read_endpoint,key,capture)
                        handles[key].add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
                for key,future in handles.items():
                    if key not in scores and future.done():
                        scores[key],gates[key]=future.result()
                        write_state(PUBLIC/'readout-progress.json',dict(scores=scores,gradient_gates=gates))
        for key,future in handles.items():
            if key not in scores:scores[key],gates[key]=future.result()
        if set(scores)!=set(plan['cases']) or set(gates)!=set(scores):raise AssertionError('Every endpoint must finish')
        if sum(r['research_optimizer_updates'] for r in records)!=plan['planned_actual_research_updates']:
            raise AssertionError('Actual shared/branched update count differs')
        documents={}
        old=root/'fixed-two-ntp-main-v1'
        for key,sha in plan['parent_document_scores_sha256'].items():
            file=old/'scores'/key/'document-scores.json'
            if file_sha256(file)!=sha:raise AssertionError('Original fitted comparison documents changed')
            documents[key]=[DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets']) for v in load(file)['records']]
        for key in scores:
            documents[key]=[DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets'])
                            for v in load(out/'scores'/key/'document-scores.json')['records']]
        comparisons={}
        for seed in [5231,17293]:
            recurrent='fixed-2b' if seed==5231 else 'fixed-s17293-data'
            for label,control in [('equal_data','one-2b' if seed==5231 else 'one-s17293-data'),
                                  ('one_work','one-work-2b' if seed==5231 else 'one-s17293-work'),
                                  ('cheap_work',f'cheap-s{seed}-work')]:
                comparisons[f'{seed}-{label}']=paired_group_interval(documents[control],documents[recurrent],
                                                                    seed=seed,weights=plan['evaluation']['weights'])
        means={label:sum(comparisons[f'{s}-{label}']['gain_nats'] for s in [5231,17293])/2
               for label in ['equal_data','one_work','cheap_work']}
        result=dict(status=COMPLETE,scores=scores,gradient_gates=gates,comparisons=comparisons,
                    descriptive_two_seed_mean_gains=means,actual_research_updates=plan['planned_actual_research_updates'],
                    training_records=records,scope=plan['scope'])
        execution.check()
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),training_records=records,scores=scores,gradient_gates=gates);raise
    finally:
        execution.failed.set();diagnostics.shutdown(wait=True,cancel_futures=True);cpu.shutdown(wait=True,cancel_futures=True)
        execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=register(a.root)
    if a.prepare_only:
        print(json.dumps(dict(status='REGISTERED_NOT_STARTED',actual_updates=plan['planned_actual_research_updates'],
                             measured_budget=plan['measured_budget'])))
    else:run(a.root,resume=a.resume)


if __name__=='__main__':main()
