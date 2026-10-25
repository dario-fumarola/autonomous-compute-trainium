"""Uniform-two-depth control using the unchanged, qualified scientific engine."""
import argparse
import concurrent.futures as cf
import copy
import json
from pathlib import Path
import signal
import time

import numpy as np
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_scientific_state import matrix_work
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY = 'fixed-two-ntp-tune-v1'
PUBLIC = Path('results/fixed_two_small_core_tune_20261025')


class GuardedExecution(Execution):
    def check(self):
        super().check(); require_inode_reserve([self.root, Path('/')])


def register(root):
    from compute_trainium.future_span import FutureSpanConfig
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from scripts.register_future_span_primitive_study import prerequisites
    if managed_running(): raise RuntimeError('Predecessor native owners must be terminal')
    predecessor = root / 'natural-chain-foundation-pilot-v1/report.json'
    if json.loads(predecessor.read_text())['status'] != 'COMPLETE_NATURAL_CHAIN_FOUNDATION_HISTORY_PILOT':
        raise ValueError('Finish the complete frozen bridge comparison first')
    _, receipts, _ = prerequisites(root)
    source_plan = root / 'future-span-tune-v2/plan.json'; source_result = root / 'future-span-tune-v2/report.json'
    old = json.loads(source_plan.read_text()); old_result = json.loads(source_result.read_text())
    if old_result['status'] != 'COMPLETE_BOUNDED_FUTURE_SPAN_TUNING' or old_result['plan_sha256'] != file_sha256(source_plan):
        raise AssertionError('Completed historical three-rate One tuning required')
    if any(file_sha256(f) != h for f, h in old['source_sha256'].items()): raise AssertionError('Qualified scientific stack changed')
    hardware = json.loads((root / 'future-span-math-qualification-v2/report.json').read_text())
    measured = next(x['native']['warm'] for x in hardware['records'] if x['case']['identity'] == 'recurrent-ntp')
    step = measured['warm_seconds']/100; cold = max(120., 1.5*(measured['wall_seconds']-measured['warm_seconds']))
    prediction = 1728*step*1.5/2 + 10*cold/2 + 4*900/3 + 3*360/3 + 300
    if prediction > 5400: raise RuntimeError('Initial conservative tuning budget does not fit')
    out = root / IDENTITY; out.mkdir(); PUBLIC.mkdir()
    cfg = ParcaeCandidateConfig(**old['model']); span = FutureSpanConfig(**old['future_config'])
    original = next(v for v in old['cases'].values() if v['cell'] == 'recurrent-ntp')
    source_table = source_plan.parent / original['depth_table']
    if file_sha256(source_table) != original['depth_table_sha256']: raise AssertionError('Original provisioned depth table changed')
    depths = np.full_like(np.load(source_table, mmap_mode='r'), 2)
    manifest = json.loads((root / 'future-span-study-preparation-v1/manifest.json').read_text())
    work = matrix_work(cfg, span, depths, world=32, optimizer_per_replica=manifest['optimizer_matrix_ops_per_replica'])
    np.save(out / 'fixed-two-depths.npy', depths); np.save(out / 'fixed-two-work.npy', work)
    if not np.all(np.load(out / 'fixed-two-depths.npy') == 2) or not np.array_equal(np.load(out / 'fixed-two-work.npy'), work):
        raise AssertionError('Immutable fixed-depth/work readback differs')
    cases = {}
    for key, old_case in old['cases'].items():
        if old_case['cell'] != 'recurrent-ntp': continue
        case = copy.deepcopy(old_case)
        case.update(depth_table='fixed-two-depths.npy', depth_table_sha256=file_sha256(out / 'fixed-two-depths.npy'),
                    work_table='fixed-two-work.npy', work_table_sha256=file_sha256(out / 'fixed-two-work.npy'),
                    training_dense_matrix_ops=int(work[:576].sum()), checkpoint_updates=[121,288,576],
                    segment_ends=[121,576], gradient_updates=[121,576])
        cases[key] = case
    if len(cases) != 3 or {c['peak_lr'] for c in cases.values()} != {.008,.016,.032}: raise AssertionError('Equal frozen tuning grid changed')
    extras = ['scripts/run_fixed_two_ntp_tune.py', 'docs/FIXED_TWO_SMALL_CORE_NTP_20261025.md']
    sources = {**old['source_sha256'], **{f:file_sha256(f) for f in extras}}
    worker, host = split_snapshot_sources(sources)
    names = [IDENTITY+'-'+key+'-'+phase for key in cases for phase in ['prefix','prefix-gradient','finish','final-gradient']]
    plan = copy.deepcopy(old)
    for stale in ['primary_interaction', 'absolute_candidate_gates', 'primary_interval_scope', 'strongest_baseline_required']:
        plan.pop(stale, None)
    plan.update(identity=IDENTITY, registered_unix=time.time(), cases=cases, source_sha256=sources,
                worker_source_sha256=worker, host_only_source_sha256=host,
                root_comm_ports=reserve_ports(root,names), seconds_limit=5400, phase_seconds_limit=1200,
                cpu_phase_seconds_limit=1200, launch_receipts=receipts,
                historical_one_tune_plan_sha256=file_sha256(source_plan), historical_one_tune_report_sha256=file_sha256(source_result),
                historical_one_tuning=[v for v in old_result['tuning_scores'] if v['cell']=='one-ntp'],
                historical_one_selection=old_result['selected']['one-ntp'],
                predecessor_report_sha256=file_sha256(predecessor), selected_rates=None,
                measured_budget=dict(reference_warm_seconds_per_update=step,cold_allowance=cold,initial_seconds_with_margin=prediction),
                primary_scope='TUNE selection only; no scientific learning or efficiency claim',
                scope=Path(extras[-1]).read_text())
    write_state(out/'plan.json',plan); write_state(PUBLIC/'plan.json',plan)


def run(root, resume=False):
    out=root/IDENTITY; plan_file=out/'plan.json'; plan=json.loads(plan_file.read_text())
    if (out/'report.json').exists(): raise FileExistsError('Preserve terminal tuning result; do not restart as new work')
    execution=GuardedExecution(root,plan_file,out/'execution',resume=resume); cpu=cf.ThreadPoolExecutor(3)
    result=dict(status='STOPPED_FIXED_TWO_NTP_TUNING'); prefixes={}; fits={}; scores={}; gates={}
    def interrupted(number,frame): execution.failed.set(); raise KeyboardInterrupt('Fixed-depth tuning interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def fit(key,phase,end,parent=None):
        name=IDENTITY+'-'+key+'-'+phase
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',key,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][name])]
        if parent:
            previous=next(x for x in parent[1]['checkpoint_records'] if x['update']==121)
            options+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/checkpoint-u121.pt','--resume-sha',previous['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_future_span_primitives',options,'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        if row['plan_sha256']!=execution.sha or row['case_id']!=key or row['update']!=end or row['qualification_only']:
            raise AssertionError('Fixed-depth fit identity differs')
        if any(not x['all_complete_replica_states_equal'] for x in row['checkpoint_records']): raise AssertionError('Scientific replicas differ')
        if parent and (not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact'] or row['recovered_state_digest']!=previous['full_state_digest']): raise AssertionError('Complete prefix restoration failed')
        if end==576:
            case=plan['cases'][key]
            if (row['real_targets'],row['valid_anchors'],row['estimated_dense_matrix_ops'])!=(case['end_targets'],case['end_anchors'],case['training_dense_matrix_ops']): raise AssertionError('Cooled endpoint counters differ')
        return name,row
    def gradient(key,capture,phase):
        name,row=capture; point=next(v for v in row['checkpoint_records'] if v['update']==row['update']); native_name=IDENTITY+'-'+key+'-'+phase
        common=['--plan-sha',execution.sha,'--case',key,'--checkpoint-sha',point['checkpoint_sha256']]
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',*common,'--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{row["update"]}.pt','--comm-port',str(plan['root_comm_ports'][native_name])]
        native=execution.native(native_name,'scripts.capture_future_span_primitives',opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        target=out/'gradients'/(key+'-'+phase)
        opts=['--root',str(root),'--plan',str(plan_file),*common,'--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),'--capture',str(root/'runs'/native_name),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        independent=execution.cpu(key+'-'+phase,'scripts.audit_future_span_gradient',opts,target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        return dict(native=native,independent_cpu=independent)
    def score(key,capture):
        name,row=capture; point=next(v for v in row['checkpoint_records'] if v['update']==576); target=out/'scores'/key
        opts=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,'--checkpoint',str(root/'runs'/name/'checkpoint-u576.pt'),'--checkpoint-sha',point['checkpoint_sha256'],'--case',key,'--cohort','tune','--depth','2','--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        return execution.cpu(key+'-score','scripts.evaluate_future_span_cpu',opts,target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_COUNTED_UNIFORM_DEPTH_PREFIXES',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as workers:
            jobs={k:workers.submit(fit,k,'prefix',121) for k in plan['cases']}
            for key,future in jobs.items(): prefixes[key]=future.result()
        warm={}
        for key,(_,row) in prefixes.items():
            points={v['update']:v for v in row['progress']}; seconds=points[120]['elapsed_seconds']-points[20]['elapsed_seconds']
            if seconds<=0: raise AssertionError('Exactly100 real warm-update timing required')
            warm[key]=dict(warm_updates=100,warm_seconds=seconds,scope='Updates21–120, actual data/optimizer/logging and runtime LR scalars, no intervening checkpoint')
        with cf.ThreadPoolExecutor(3) as workers:
            jobs={k:workers.submit(gradient,k,capture,'prefix-gradient') for k,capture in prefixes.items()}
            for key,future in jobs.items(): gates[key]=future.result()
        rate=max(v['warm_seconds']/100 for v in warm.values()); remaining=3*455*rate*1.5/2+4*plan['measured_budget']['cold_allowance']/2+900+360+300
        readiness=dict(actual_warm=warm,predicted_remaining_seconds_with_margin=remaining,seconds_remaining=execution.deadline-time.time())
        write_state(PUBLIC/'readiness.json',readiness)
        if remaining>readiness['seconds_remaining']: raise RuntimeError('Remaining complete frozen tuning stages do not fit original cap')
        write_state(PUBLIC/'status.json',dict(status='RUNNING_COOLED_UNIFORM_DEPTH_TUNING',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as workers:
            jobs={k:workers.submit(fit,k,'finish',576,prefixes[k]) for k in plan['cases']}
            for key,future in jobs.items(): fits[key]=future.result()
        pending={k:cpu.submit(score,k,capture) for k,capture in fits.items()}
        for key,future in pending.items(): scores[key]=future.result()
        selected=min(scores,key=lambda k:(scores[k]['metrics']['weighted_nll'],plan['cases'][k]['peak_lr']))
        final_gate=gradient(selected,fits[selected],'final-gradient')
        result=dict(status='COMPLETE_FIXED_TWO_NTP_TUNING',selected_case=selected,selected_rate=plan['cases'][selected]['peak_lr'],boundary_winner=plan['cases'][selected]['peak_lr'] in [.008,.032],scores=scores,prefix_gates=gates,selected_final_gate=final_gate,warm=warm,completed_fit_reports={k:v[1] for k,v in fits.items()},historical_one_selection=plan['historical_one_selection'])
    except BaseException as error:
        execution.failed.set(); result.update(error=str(error),prefix_reports={k:v[1] for k,v in prefixes.items()},completed_fit_reports={k:v[1] for k,v in fits.items()},scores=scores,prefix_gates=gates); raise
    finally:
        execution.failed.set(); cpu.shutdown(wait=True,cancel_futures=True); execution.close(PUBLIC/'execution-costs.json')
        for s,handler in handlers.items(): signal.signal(s,handler)
        result.update(plan_sha256=execution.sha,scope=plan['scope'],test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result); write_state(PUBLIC/'report.json',result); write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not (a.root/IDENTITY/'plan.json').exists(): register(a.root)
    if not a.prepare_only: run(a.root,a.resume)


if __name__=='__main__': main()
