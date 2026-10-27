"""Explicit untouched remaining opportunities after an observed audit rejection."""
import argparse
import json
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_scale_boundary_lower_rates as engine

IDENTITY='scale-boundary-last-original-rates-v1'
PUBLIC=Path('results/scale_boundary_last_original_rates_20261027')
DOCUMENT='docs/SCALE_LAST_ORIGINAL_RATES_20261027.md'


def rejection(root):
    folder=root/'scale-r032-audit-observation-v1'
    result=json.loads((folder/'report.json').read_text())
    metrics=json.loads((folder/'audit/full-gradient-observation.json').read_text())
    scope=json.loads((folder/'plan.json').read_text())
    if (result['status']!='OBSERVED_ORIGINAL_FULL512_NUMERICAL_REJECTION'
            or metrics['windows']!=512 or not all(metrics[k] for k in [
                'signature_matches','source_cursor_matches','real_targets_match','native_clipped','model_unchanged'])
            or (metrics['gradient']['passes_two_percent_gate'] and abs(metrics['native_minus_cpu_nll'])<=.003)
            or scope['original_plan_sha256']!=file_sha256(root/engine.PARENT/'plan.json')):
        raise ValueError('Actual unchanged full512 numerical rejection required')
    return dict(declared_numerically_ineligible=True,
                reason='Original mature512 numerical threshold failed; no admissible quality score',
                original_audit_failure_sha256=file_sha256(root/engine.PARENT/'audits/fixed-two-lr2/failure.json'),
                repeated_observation_sha256=file_sha256(folder/'audit/full-gradient-observation.json'),
                observation_report_sha256=file_sha256(folder/'report.json'))


def prepare(root):
    if managed_running():raise RuntimeError('No admission over active Native execution')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every completion attempt')
    rejected=rejection(root);parent_file=root/engine.PARENT/'plan.json'
    parent=json.loads(parent_file.read_text());result_file=root/engine.PARENT/'report.json'
    result=json.loads(result_file.read_text())
    if (result['status']!='STOPPED_BOUNDARY_RATE_COMPLETION'
            or result.get('error')!='CPU phase failed: fixed-two-lr2-audit'
            or result['plan_sha256']!=file_sha256(parent_file)):
        raise ValueError('Exact stopped parent required')
    amendment=root/'research-clock-accounting-amendment-20261027-v1/plan.json'
    clock=json.loads(amendment.read_text())
    if not clock['overall_clock_admission_waived'] or not clock['scientific_endpoints_unchanged']:
        raise ValueError('Explicit owner clock amendment required')
    if shutil.disk_usage(root).free<53*1024**3:raise RuntimeError('New18GiB diagnostic/failure peak plus35GiB reserve required')
    if shutil.disk_usage(root.parent/'recurrence').free<39*1024**3:raise RuntimeError('Both retained predictions plus reserve required')
    for name,sha in parent['source_sha256'].items():
        if (name.startswith('compute_trainium/') or name in ['scripts/train_scale_registered.py',
                'scripts/train_scale_capture_invalid_gradient.py','scripts/audit_scale_registered.py',
                'scripts/score_scale_registered.py']) and file_sha256(name)!=sha:
            raise AssertionError('Qualified original mathematical source changed:'+name)
    for name,sha in parent['numerical_gates'].items():
        if file_sha256(root/name/'completion-verification.json')!=sha:raise AssertionError('Original numerical proof changed')
    original=json.loads((root/'scale-medium-tune-65536-v1/plan.json').read_text())
    order=['ordinary8-lr3','fixed-two-lr3'];cases={};phases={};names=[];limits={}
    for key in order:
        case=dict(parent['cases'][key]);artifact=IDENTITY+'/artifacts/'+key
        if file_sha256(root/case['initializer'])!=case['initializer_sha256']:raise AssertionError('Untouched original C0 required')
        case.update(artifact_directory=artifact+'/warm',replay_directory=artifact+'/replay',
                    audit_directory=IDENTITY+'/audits/'+key,score_directory=IDENTITY+'/scores/'+key,
                    retained_model='../recurrence/'+IDENTITY+'/retained/'+key+'.pt',cleanup_receipt=IDENTITY+'/cleanup/'+key+'.json')
        cases[key]=case;cases[key+'-replay']={**case,'artifact_directory':case['replay_directory'],
            'resume':dict(directory=case['artifact_directory'],update=299),'save_updates':[300],
            'capture_before_updates':[],'audit_first_update':False}
        for suffix,ck,seconds in [('warm',key,1500),('replay',key+'-replay',600)]:
            name=IDENTITY+'-'+key+'-'+suffix;names.append(name);phases[name]=dict(case=ck);limits[name]=seconds
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party']
             for f in Path(d).rglob('*.py') if f.is_file()}
    for name in ['scripts/run_native_isolated.sh',DOCUMENT]:sources[name]=file_sha256(name)
    worker,host=split_snapshot_sources(sources);start=time.time()
    plan={**parent,'identity':IDENTITY,'registered_unix':start,'source_sha256':sources,
        'worker_source_sha256':worker,'host_only_source_sha256':host,
        'clock_admission_waiver':True,'user_clock_amendment_sha256':file_sha256(amendment),
        'seconds_limit':4*3600,'absolute_deadline_unix':start+4*3600,
        'original_stopped_parent_plan_sha256':file_sha256(parent_file),
        'parent_report_sha256':file_sha256(result_file),'selection_parent_report':str(result_file.relative_to(root)),
        'declared_ineligible_records':{'fixed-two-lr2':{**result['records']['fixed-two-lr2'],**rejected}},
        'selection_rates':{key:original['cases'][key]['peak_lr'] for key in original['case_order']},
        'selection_expected_per_role':4,'selection_rate_grid':[.008,.016,.032,.064],
        'complete_selection_status':'COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION',
        'stopped_selection_status':'STOPPED_LAST_ORIGINAL_RATE_COMPLETION',
        'audit_entry_module':'scripts.audit_scale_failure_metrics',
        'cases':cases,'case_order':order,'phases':phases,'native_phase_seconds_limits':limits,
        'root_comm_ports':reserve_ports(root,names),
        'case_native_phases':{k:[IDENTITY+'-'+k+'-'+s for s in ['warm','replay']] for k in order},
        'allowed_tune_artifact_roots':[IDENTITY+'/artifacts'],'actual_new_tune_update_allowance':2*301,
        'scope':Path(DOCUMENT).read_text(),'completion_scope_sha256':file_sha256(DOCUMENT),
        'scientific_main_fit_admitted':False}
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',action='store_true');p.add_argument('--then-lower',action='store_true');a=p.parse_args()
    if not a.run:prepare(a.root)
    else:
        engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.run(a.root)
        if a.then_lower:
            engine.IDENTITY='scale-boundary-lower-rates-v1'
            engine.PUBLIC=Path('results/scale_boundary_lower_rates_20261027')
            if engine.prepare(a.root):engine.run(a.root)
