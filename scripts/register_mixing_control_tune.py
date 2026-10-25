"""Register three bounded fair LR opportunities after complete hardware pass."""
import copy
import json
import time
from pathlib import Path

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.mixing_control_science import matrix_work,VARIANT
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='small-core-mixing-tune-v1'
PUBLIC=Path('results/small_core_mixing_tune_20261025')


def register(root):
    hardware_file=root/'small-core-mixing-qualification-v1/report.json'
    hardware=json.loads(hardware_file.read_text())
    if hardware['status']!='PASS_COMPLETE_CHEAP_MIXING_CONTROL_QUALIFICATION' or managed_running():
        raise RuntimeError('Complete independent qualification and idle native ownership required')
    qual_file=root/'small-core-mixing-qualification-v1/plan.json'
    qualification=json.loads(qual_file.read_text())
    old_file=root/'future-span-tune-v2/plan.json';old=json.loads(old_file.read_text())
    geometry_file=Path('results/small_core_mixing_control_preparation_20261025/prediction-geometry.json')
    geometry=json.loads(geometry_file.read_text())
    if geometry['status']!='PASS_CPU_CONTROL_PREDICTION_AND_GEOMETRY_ONLY':
        raise ValueError('Independent prediction path and work geometry required')
    additions=['compute_trainium/mixing_control_science.py','tests/test_mixing_control_science.py',
        'scripts/train_mixing_control_scientific.py','scripts/evaluate_mixing_control_cpu.py',
        'scripts/capture_mixing_control_gradient.py','scripts/audit_mixing_scientific_gradient.py',
        'scripts/register_mixing_control_tune.py','scripts/run_mixing_control_tune.py',
        'docs/SMALL_CORE_MIXING_CONTROL_PROTOCOL_20261025.md']
    sources={**old['source_sha256'],**qualification['source_sha256'],**{p:file_sha256(p) for p in additions}}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise AssertionError('Frozen source changed')
    worker,host=split_snapshot_sources(sources)
    measured=next(r['native']['warm'] for r in hardware['records'] if r['case']['arm']=='recurrent')
    if measured['warm_updates']!=100:raise ValueError('Actual100-update throughput required')
    step=measured['warm_seconds']/100
    cold=max(0,measured['wall_seconds']-measured['warm_seconds'])
    # Two waves per phase:preflight,capture,finish,capture. CPU gates/tails
    # are separately reserved;no inference from a few printed updates.
    estimate=2*576*step*1.35+8*(cold+20)+900
    if not 0<estimate<5400:raise RuntimeError('Measured complete TUNE budget does not fit')
    out=root/IDENTITY;out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    c=ParcaeCandidateConfig(**old['model']);span=FutureSpanConfig(**old['future_config'])
    sample=next(v for v in old['cases'].values() if v['cell']=='recurrent-ntp')
    depth_file=old_file.parent/sample['depth_table']
    if file_sha256(depth_file)!=sample['depth_table_sha256']:raise AssertionError('Original mixing schedule changed')
    depths=np.load(depth_file,mmap_mode='r')
    work=matrix_work(c,span,depths,world=32,optimizer_per_replica=geometry['operator_ledger_per_replica_optimizer'])
    np.save(out/'mixing-work.npy',work)
    cases={}
    for key,value in old['cases'].items():
        if value['cell']!='recurrent-ntp':continue
        case=copy.deepcopy(value)
        case.update(depth_table='../future-span-tune-v2/'+value['depth_table'],work_table='mixing-work.npy',
            work_table_sha256=file_sha256(out/'mixing-work.npy'),training_dense_matrix_ops=int(work[:576].sum()),
            checkpoint_updates=[2,288,576],segment_ends=[2,576],gradient_updates=[2,576])
        cases[key]=case
    if len(cases)!=3 or sorted(c['peak_lr'] for c in cases.values())!=[.008,.016,.032]:
        raise AssertionError('Exactly the original three independent opportunities required')
    names=[IDENTITY+'-'+k+'-'+phase for k in cases for phase in ['preflight','preflight-gradient','finish','final-gradient']]
    plan=copy.deepcopy(old)
    plan.update(identity=IDENTITY,registered_unix=time.time(),stage='cheap_control_tune',model_variant=VARIANT,
        cases=cases,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),seconds_limit=5400,phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,
        hardware_report_sha256=file_sha256(hardware_file),hardware_plan_sha256=file_sha256(qual_file),
        geometry_report_sha256=file_sha256(geometry_file),original_tune_plan_sha256=file_sha256(old_file),
        measured_budget={'warm_step_seconds':step,'cold_seconds':cold,'conservative_seconds':estimate,'cap_seconds':5400},
        scope=Path('docs/SMALL_CORE_MIXING_CONTROL_PROTOCOL_20261025.md').read_text())
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan
