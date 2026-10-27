"""One frozen joint-scale development pair and a completed equal-data branch."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import AllChipExecution,SCALE_START,SCALE_END

IDENTITY='scale-medium-main-v1'
PUBLIC=Path('results/scale_medium_main_20261027')


def boundary_scope_files(root,*,body_exposure=False):
    original=root/'scale-boundary-scope-freeze-v1/scope.json'
    revised=root/'scale-body-exposure-scope-v1/scope.json'
    if revised.exists() and not body_exposure:
        raise ValueError('Unadmitted old main superseded by the explicit body-exposure scope')
    chosen=revised if body_exposure else original
    frozen=json.loads(chosen.read_text())
    if body_exposure and frozen['original_unadmitted_boundary_scope_sha256']!=file_sha256(original):
        raise AssertionError('Original precision/source scope changed')
    document='docs/SCALE_BODY_EXPOSURE_DEVELOPMENT_20261027.md' if body_exposure else 'docs/SCALE_BOUNDARY_RATE_COMPLETION_20261027.md'
    if file_sha256(document)!=frozen['scope_sha256']:
        raise AssertionError('Prospectively frozen scientific scope changed')
    return original,chosen,document,frozen


def validate_storage_binding(proof,*,tuning_sha256,scope_sha256,boundary_proof_sha256):
    if (proof['tuning_report_sha256']!=tuning_sha256 or proof['scope_freeze_sha256']!=scope_sha256
            or proof['boundary_numerical_proof_sha256']!=boundary_proof_sha256):
        raise AssertionError('Storage receipt belongs to a different tuning result,scope or numerical proof')
    if (proof['protected_host_bytes']!=512*1024**3 or proof['protected_filesystem_bytes']!=35*1024**3
            or proof['native_worker_allowance_bytes']!=192*1024**3 or proof['new_ram_allowance_bytes']!=0):
        raise AssertionError('Storage receipt changed the qualified resource reserves')


def validate_body_scope_inputs(frozen,*,manifest_sha256,packing_fingerprint,initializer_report_sha256,clock_sha256):
    if (frozen['accepted_packing_manifest_sha256']!=manifest_sha256
            or frozen['packing_fingerprint']!=packing_fingerprint
            or frozen['original_unused_initializer_report_sha256']!=initializer_report_sha256
            or frozen['user_clock_amendment_sha256']!=clock_sha256):
        raise AssertionError('Frozen accepted packing,unused C0 report or clock binding changed')


def prepare(root,*,boundary_corrected=False,body_exposure=False):
    import numpy as np
    from compute_trainium.scale_pilot_size import profiled_inventory
    from compute_trainium.scale_work_ledger import fixed_update_work,work_matched_updates
    if managed_running():raise RuntimeError('Complete prior Native closure required')
    inode_projection=project_snapshot_inodes(Path('.'),6)
    if os.statvfs(root).f_favail<inode_projection['required_free_inodes']:
        raise RuntimeError('All six main snapshots and artifacts must fit the protected inode reserve')
    tune_identity='scale-boundary-rate-completion-v2' if boundary_corrected else 'scale-medium-tune-65536-v1'
    tune_file=root/tune_identity/'report.json' if boundary_corrected else root/'scale-tune-readout-recovery-v1/report.json'
    if boundary_corrected:
        from scripts.run_scale_boundary_lower_rates import selection_source
        tune_plan_file,tune_file=selection_source(root)
        tune_identity=tune_plan_file.parent.name
    tune=json.loads(tune_file.read_text())
    expected=('COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION' if tune_identity in ['scale-boundary-lower-rates-v1','scale-final-lower-rate-recovery-v1']
              else 'COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION' if tune_identity=='scale-boundary-last-original-rates-v1'
              else 'COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION') if boundary_corrected else 'COMPLETE_INDEPENDENT_SCALE_RATE_SELECTION'
    if tune['status']!=expected:raise ValueError('Complete independently audited corrected TUNE required')
    tune_plan=json.loads((root/tune_identity/'plan.json').read_text())
    if tune['plan_sha256']!=file_sha256(root/tune_identity/'plan.json'):raise AssertionError('Original TUNE registration changed')
    if body_exposure and not boundary_corrected:raise ValueError('Body-exposure scope requires qualified corrected boundaries')
    original_scope_file=root/'scale-boundary-scope-freeze-v1/scope.json'
    scope_file=original_scope_file;scope_document='docs/SCALE_JOINT_DEVELOPMENT_20261027.md';frozen=None
    if boundary_corrected:
        original_scope_file,scope_file,scope_document,frozen=boundary_scope_files(root,body_exposure=body_exposure)
    if boundary_corrected and (tune_plan['scope_freeze_sha256']!=file_sha256(original_scope_file)
            or tune_plan.get('precision_policy')!='fp32_recurrent_boundaries'):
        raise AssertionError('Outcome-blind boundary scope and qualified policy required')
    r_end=frozen['recurrent_updates'] if frozen else 9554
    o_end=frozen['ordinary_work_updates'] if frozen else 12939
    r_stable=frozen['stable_trunk_update'] if frozen else 8598
    o_stable=frozen['ordinary_work_stable_update'] if frozen else 11645
    main_seconds=frozen['main_required_seconds'] if frozen else 14*3600
    clock_amendment=root/'research-clock-accounting-amendment-20261027-v1/plan.json'
    clock=json.loads(clock_amendment.read_text()) if boundary_corrected else None
    if clock and (not clock['overall_clock_admission_waived'] or not clock['scientific_endpoints_unchanged']):
        raise ValueError('Explicit unchanged-endpoint owner clock amendment required')
    forecast_seconds=main_seconds
    if clock:main_seconds=clock['operational_main_watchdog_seconds']
    if boundary_corrected:
        if file_sha256(scope_document)!=frozen['scope_sha256']:raise AssertionError('Frozen scope document changed')
        for name,sha in tune_plan['source_sha256'].items():
            if (name.startswith('compute_trainium/') or name in ['scripts/train_scale_registered.py','scripts/train_scale_capture_invalid_gradient.py','scripts/audit_scale_registered.py','scripts/score_scale_registered.py']) and file_sha256(name)!=sha:
                raise AssertionError('Tuned mathematical/training/readout source changed:'+name)
    if tune_plan.get('manual_loss_scale')!=65536:raise ValueError('Both roles must retain the freshly qualified scaler')
    for role in ['fixed_two','ordinary8']:
        choice=tune['selected'][role]
        allowed=tune_plan.get('selection_rate_grid',[.008,.016,.032,.064])
        if choice['rate'] not in allowed or choice['score']>=4.:raise ValueError('Prespecified healthy independent rate required')
    storage=root/('scale-body-exposure-main-state-allowance-v1' if body_exposure else 'scale-boundary-main-state-allowance-v1' if boundary_corrected else 'scale-main-state-allowance-v2')/'report.json'
    storage_proof=json.loads(storage.read_text())
    if storage_proof['status']!='ADMITTED_COMPLETE_SCALE_MAIN_STATE_STORAGE':raise ValueError('Complete prospective raw-state storage required')
    if boundary_corrected:
        validate_storage_binding(storage_proof,tuning_sha256=file_sha256(tune_file),scope_sha256=file_sha256(scope_file),
            boundary_proof_sha256=file_sha256(root/'scale-boundary-fixed-two-numerical-v1/completion-verification.json'))
    if shutil.disk_usage(root).free<35*1024**3+storage_proof['complete_additional_bytes']:
        raise RuntimeError('Complete additional state allowance must still fit at actual main admission')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every main registration')
    start=time.time()
    if not clock and start+main_seconds>SCALE_END:raise TimeoutError('Complete measured main allowance does not fit the original scale cap')
    initial=root/'scale-main-original-initializers-v1';initial_report=json.loads((initial/'report.json').read_text())
    if (initial_report['phase']!='main' or initial_report['seed']!=49177
            or initial_report['optimizer_updates']!=0 or initial_report['model_forwards']!=0):raise ValueError('Fresh original main C0 required')
    inventory=profiled_inventory('medium');data=root/'natural-append-capacity-v4'
    packed=json.loads((data/'packed/manifest.json').read_text());counts=np.load(data/'packed/prefix_counts.npy',mmap_mode='r')
    if body_exposure:
        validate_body_scope_inputs(frozen,manifest_sha256=file_sha256(data/'packed/manifest.json'),
            packing_fingerprint=packed['fingerprint'],initializer_report_sha256=file_sha256(initial/'report.json'),
            clock_sha256=file_sha256(clock_amendment))
    def targets(u):return sum(map(int,counts[u*512]))
    rwork=fixed_update_work(inventory[0]['config'],'fixed_two')['total_per_update']
    owork=fixed_update_work(inventory[1]['config'],'ordinary8')['total_per_update']
    matching=work_matched_updates(rwork,r_end,owork)
    expected_targets=(frozen['recurrent_targets'],frozen['ordinary_work_targets']) if frozen else (5000260971,6771820945)
    if matching['updates']!=o_end or (targets(r_end),targets(o_end))!=expected_targets:
        raise AssertionError('Prospectively fixed data/work endpoints changed')
    cases={};phases={};names=[];limits={}
    def base(role,end,stable,lineage_case):
        row=next(r for r in inventory if r['role']==role);record=next(r for r in initial_report['records'] if r['role']==role)
        if file_sha256(initial/record['file'])!=record['sha256']:raise AssertionError('Original main initializer changed')
        return dict(phase='main',seed=49177,state_seed=49210 if role=='fixed_two' else None,
            model_role=role,model=row['config'],parameters=row['parameters'],initializer='scale-main-original-initializers-v1/'+record['file'],
            initializer_sha256=record['sha256'],packing_fingerprint=packed['fingerprint'],peak_lr=tune['selected'][role]['rate'],
            lineage_case=lineage_case,endpoint_updates=end,stop_update=end,warmup_targets=100000000,
            cooldown_targets=targets(stable),end_targets=targets(end),work_per_update=rwork if role=='fixed_two' else owork,
            audit_first_update=False,save_updates=[end-8,end],capture_before_updates=[],cohort='curve',retained_model=None,
            post_training_probe=True,audit_model_update=end,replay_before_update=end-8,periodic_checkpoint_interval=1000)
    def add(key,case,seconds):
        cases[key]=case;name=IDENTITY+'-'+key;phases[name]=dict(case=key);names.append(name);limits[name]=seconds
    recurrent=base('fixed_two',r_end,r_stable,'recurrent-main')
    recurrent.update(artifact_directory=IDENTITY+'/artifacts/recurrent/warm',replay_directory=IDENTITY+'/artifacts/recurrent/replay',
        audit_directory=IDENTITY+'/audits/recurrent',score_directory=IDENTITY+'/scores/recurrent')
    phase_limits=frozen['phase_limits'] if body_exposure else dict(recurrent=8*3600,ordinary_trunk=5*3600,ordinary_equal=3600,ordinary_finish=3*3600)
    add('recurrent',recurrent,phase_limits['recurrent'])
    add('recurrent-replay',{**recurrent,'artifact_directory':recurrent['replay_directory'],
        'resume':dict(directory=recurrent['artifact_directory'],update=r_end-8),'save_updates':[r_end],
        'capture_before_updates':[],'post_training_probe':False,'periodic_checkpoint_interval':0},600)
    ordinary=base('ordinary8',o_end,o_stable,'ordinary-main')
    ordinary.update(artifact_directory=IDENTITY+'/artifacts/ordinary/warm',replay_directory=IDENTITY+'/artifacts/ordinary/replay',
        audit_directory=IDENTITY+'/audits/ordinary',score_directory=IDENTITY+'/scores/ordinary')
    trunk={**ordinary,'stop_update':r_stable,'save_updates':[r_stable],'capture_before_updates':[],
        'artifact_directory':IDENTITY+'/artifacts/ordinary/trunk','post_training_probe':False}
    add('ordinary-trunk',trunk,phase_limits['ordinary_trunk'])
    equal=base('ordinary8',r_end,r_stable,'ordinary-main')
    equal.update(artifact_directory=IDENTITY+'/artifacts/ordinary-equal/warm',
        audit_directory=IDENTITY+'/audits/ordinary-equal',score_directory=IDENTITY+'/scores/ordinary-equal',
        resume=dict(directory=trunk['artifact_directory'],update=r_stable))
    add('ordinary-equal',equal,phase_limits['ordinary_equal'])
    add('ordinary',{**ordinary,'resume':dict(directory=trunk['artifact_directory'],update=r_stable)},phase_limits['ordinary_finish'])
    add('ordinary-replay',{**ordinary,'artifact_directory':ordinary['replay_directory'],
        'resume':dict(directory=ordinary['artifact_directory'],update=o_end-8),'save_updates':[o_end],
        'capture_before_updates':[],'post_training_probe':False,'periodic_checkpoint_interval':0},600)
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for f in [scope_document,'tests/test_scale_training_state.py','tests/test_scale_training_fixture.py',
              'tests/test_scale_work_ledger.py','scripts/run_native_isolated.sh']:sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);deadline=start+main_seconds if clock else min(start+main_seconds,SCALE_END)
    if shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected main NVMe reserve')
    folder.mkdir();PUBLIC.mkdir()
    plan=dict(identity=IDENTITY,purpose='scale_registered_development_v1',registered_unix=start,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        world=64,worker_memory_gib=192,inode_projection=inode_projection,seconds_limit=main_seconds,phase_seconds_limit=8*3600,
        native_phase_seconds_limits=limits,cpu_phase_seconds_limit=1800,readout_seconds_limit=1800,
        absolute_deadline_unix=deadline,scale_budget_started_unix=SCALE_START,scale_budget_deadline_unix=SCALE_END,
        trainer_gate_receipt=tune_plan['trainer_gate_receipt'],trainer_gate_sha256=tune_plan['trainer_gate_sha256'],
        numerical_gates=tune_plan['numerical_gates'],data=tune_plan['data'],evaluation=tune_plan['evaluation'],
        tuning_plan_sha256=tune['plan_sha256'],tuning_report_sha256=file_sha256(tune_file),
        independently_selected_rates=tune['selected'],initializer_report_sha256=file_sha256(initial/'report.json'),
        storage_admission_sha256=file_sha256(storage),manual_loss_scale=65536,
        precision_policy='fp32_recurrent_boundaries' if boundary_corrected else 'fp16_body',invalid_gradient_capture=boundary_corrected,
        scope_freeze_sha256=file_sha256(scope_file) if boundary_corrected else None,
        original_precision_scope_freeze_sha256=file_sha256(original_scope_file) if boundary_corrected else None,
        prospective_body_exposure_scope=body_exposure,
        clock_admission_waiver=bool(clock),user_clock_amendment_sha256=file_sha256(clock_amendment) if clock else None,
        training_time_forecast_seconds=forecast_seconds,scientific_endpoints_unchanged_by_clock_amendment=True,
        periodic_storage_lifecycle='two_latest_periodic_until_verified_permanent_successor',
        cases=cases,phases=phases,root_comm_ports=reserve_ports(root,names),
        native_order=['recurrent','recurrent-replay','ordinary-trunk','ordinary-equal','ordinary','ordinary-replay'],
        equal_data_branch_complete_cooldown=True,work_match=matching,minimum_useful_gain=.01,
        initial_training_seed_pairs=1,consumed_development_only=True,terminal_alpha_spent=0,
        scientific_main_fit_admitted=True,no_automatic_confirmation_or_followup=True,
        scope=Path(scope_document).read_text())
    write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_MAIN_NOT_ACTIVATED',terminal_claim=False))


def analyze(root,plan_sha,records):
    from compute_trainium.metrics import DocumentScore,paired_group_interval
    def rows(key):
        value=json.loads((root/IDENTITY/'scores'/key/'document-scores.json').read_text())['records']
        return [DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets']) for v in value]
    recurrent=rows('recurrent');primary=paired_group_interval(rows('ordinary'),recurrent,draws=10000,seed=49177)
    equal=paired_group_interval(rows('ordinary-equal'),recurrent,draws=10000,seed=49177)
    return dict(status='COMPLETE_ONE_PAIR_SCALE_DEVELOPMENT_COMPARISON',plan_sha256=plan_sha,
        primary_recurrent_gain_over_ordinary_work=primary,secondary_equal_data_gain=equal,
        one_training_seed_only=True,consumed_development_panel=True,terminal_claim=False,
        development_mean_meets_point_threshold=primary['gain_nats']>=.01,
        decision='Potential development candidate only; fresh adequately powered confirmation and inference costs remain required.' if primary['gain_nats']>=.01 else
            'Does not advance at the declared useful-effect floor against this conventional challenger; no automatic extra exposure or mechanism.',
        warning='Source-group interval is conditional on this trained pair, not training-seed uncertainty. Nominal matrix work is not chip-time matching.')


def run(root):
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');records={}
    result=dict(status='STOPPED_SCALE_MAIN',records=records,terminal_claim=False)
    def stop(number,frame):raise KeyboardInterrupt('Bounded scale main interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    def evaluate(key):
        case=execution.plan['cases'][key]
        record=records[key]
        record['audit']=execution.cpu(key+'-audit','scripts.audit_scale_registered',
            ['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['audit_directory'])],
            root/case['audit_directory']/'report.json','PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
        record['score']=execution.cpu(key+'-score','scripts.score_scale_registered',
            ['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['score_directory'])],
            root/case['score_directory']/'report.json','COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
    try:
        for key in execution.plan['native_order']:
            name=IDENTITY+'-'+key
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                '--plan-sha',execution.sha,'--case',key,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            entry='scripts.train_scale_capture_invalid_gradient' if execution.plan.get('invalid_gradient_capture') else 'scripts.train_scale_registered'
            records[key]=dict(native=execution.native(name,entry,options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE'))
            if key=='recurrent-replay':evaluate('recurrent')
            elif key=='ordinary-equal':evaluate('ordinary-equal')
            elif key=='ordinary-replay':evaluate('ordinary')
            write_state(folder/'progress.json',result)
        result.update(analysis=analyze(root,execution.sha,records),status='COMPLETE_SCALE_JOINT_DEVELOPMENT')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(plan_sha256=execution.sha);execution.close(folder/'execution-costs.json')
        write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],terminal_claim=False))
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
