"""Separately budget the prespecified two-seed cheap-control branches."""
import copy
import json
import time
from pathlib import Path

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='small-core-mixing-main-v1'
PUBLIC=Path('results/small_core_mixing_main_20261025')


def register(root):
    tune_file=root/'small-core-mixing-tune-v2/report.json'
    tune=json.loads(tune_file.read_text())
    source_file=root/'small-core-mixing-tune-v2/plan.json'
    source=json.loads(source_file.read_text())
    if (tune['status']!='COMPLETE_BOUNDED_CHEAP_MIXING_TUNE' or len(tune['scores'])!=3
            or len(tune['gradient_gates'])!=6 or tune['plan_sha256']!=file_sha256(source_file)
            or managed_running()):
        raise RuntimeError('All three tuning opportunities and full numerical gates required')
    winner=min(tune['scores'],key=lambda s:(s['metrics']['weighted_nll'],s['peak_lr']))
    if winner['peak_lr']!=tune['selected_lr']:
        raise AssertionError('Frozen tuning selection disagrees')
    original_file=root/'future-span-main-v2/plan.json'
    original=json.loads(original_file.read_text())
    prior_file=root/'future-span-main-v2/report.json'
    work_file=root/'future-span-work-control-v1/report.json'
    measured=json.loads((root/'small-core-mixing-qualification-v1/report.json').read_text())
    row=next(r['native']['warm'] for r in measured['records'] if r['case']['arm']=='recurrent')
    step=row['warm_seconds']/100;cold=row['wall_seconds']-row['warm_seconds']
    # Each paired seed takes960 stable updates,then two independent960/1387
    # continuations. Two halves run paired seeds;four endpoint branches follow.
    estimate=3307*step*1.35+8*(cold+20)+1200
    if not 0<estimate<10800:raise RuntimeError('Measured complete MAIN cannot fit its3-hour cap')
    additions=['scripts/register_mixing_control_main.py','scripts/run_mixing_control_main.py']
    sources={**source['source_sha256'],**{p:file_sha256(p) for p in additions}}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise AssertionError('Exact original sources changed')
    worker,host=split_snapshot_sources(sources)
    source_case=next(iter(source['cases'].values()))
    depth_file=source_file.parent/source_case['depth_table']
    work_table=source_file.parent/source_case['work_table']
    if file_sha256(work_table)!=source_case['work_table_sha256'] or file_sha256(depth_file)!=source_case['depth_table_sha256']:
        raise AssertionError('Original cheap-control work/depth tables changed')
    geometry=json.loads(Path('results/small_core_mixing_control_preparation_20261025/prediction-geometry.json').read_text())
    table=np.load(work_table,mmap_mode='r')
    counts=np.load(root/source['data']['packed']/'prefix_counts.npy',mmap_mode='r')
    aux=source['auxiliary_data'];anchors=np.load(root/aux['directory']/aux['prefix_file'],mmap_mode='r')
    cases={}
    for seed in [5231,7283]:
        old=original['cases'][f'recurrent-ntp-s{seed}-lr32000']
        short_key=f'recurrent-ntp-s{seed}-equal-data'
        for label,end in [('equal-data',1920),('matched-work',2347)]:
            key=f'recurrent-ntp-s{seed}-{label}'
            case=copy.deepcopy(old);cool=end//2
            case.update(peak_lr=winner['peak_lr'],total_updates=end,end_targets=int((counts[end*512]-counts[0]).sum()),
                end_anchors=int(anchors[end*512]),training_dense_matrix_ops=int(table[:end].sum()),
                cooldown_start_update=cool,cooldown_start_targets=int((counts[cool*512]-counts[0]).sum()),
                depth_table='../small-core-mixing-tune-v2/'+source_case['depth_table'],
                work_table='../small-core-mixing-tune-v2/'+source_case['work_table'],
                work_table_sha256=source_case['work_table_sha256'],
                checkpoint_updates=[2,960,end] if label=='equal-data' else [end],
                segment_ends=[2,960,end] if label=='equal-data' else [end],
                gradient_updates=[2,end] if label=='equal-data' else [end])
            if label=='matched-work':case['parent']={'case_id':short_key,'update':960}
            expected=geometry['cheap_equal_data' if label=='equal-data' else 'cheap_matched_candidate_work']
            if case['end_targets']!=expected['real_targets'] or case['training_dense_matrix_ops']!=expected['complete_matrix_work']:
                raise AssertionError('Prespecified endpoint geometry changed')
            cases[key]=case
    phases={}
    for key,c in cases.items():
        labels=['preflight','preflight-gradient','stable','endpoint','final-gradient'] if key.endswith('equal-data') else ['endpoint','final-gradient']
        phases.update({IDENTITY+'-'+key+'-'+phase:phase for phase in labels})
    plan=copy.deepcopy(source)
    plan.update(identity=IDENTITY,registered_unix=time.time(),stage='cheap_control_main',cases=cases,
        initial_states=original['initial_states'],source_sha256=sources,worker_source_sha256=worker,
        host_only_source_sha256=host,root_comm_ports=reserve_ports(root,list(phases)),seconds_limit=10800,
        phase_seconds_limit=3600,cpu_phase_seconds_limit=1200,
        tune_report_sha256=file_sha256(tune_file),tune_plan_sha256=file_sha256(source_file),
        original_main_plan_sha256=file_sha256(original_file),original_main_report_sha256=file_sha256(prior_file),
        original_work_report_sha256=file_sha256(work_file),selected_lr=winner['peak_lr'],
        measured_budget={'warm_step_seconds':step,'cold_seconds':cold,'conservative_seconds':estimate,'cap_seconds':10800})
    out=root/IDENTITY;out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan
