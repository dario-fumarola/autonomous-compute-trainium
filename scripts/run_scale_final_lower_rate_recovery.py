"""Complete only the final ordinary rate after the preserved inode stop."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_scale_boundary_lower_rates as engine

IDENTITY=engine.FINAL_RECOVERY
PUBLIC=Path('results/scale_final_lower_rate_recovery_20261027')
DOCUMENT='docs/SCALE_FINAL_LOWER_RATE_RECOVERY_20261027.md'
CASE='ordinary8-lower-lr1'


def validate_stop(root,parent,stopped):
    if (stopped['status']!='STOPPED_BOUNDARY_LOWER_RATES'
            or stopped.get('error')!='Filesystem inode reserve failed: '+str(root)
            or stopped['records'].get(CASE)!={}
            or parent['case_order']!=['fixed-two-lower-lr0','ordinary8-lower-lr0','fixed-two-lower-lr1',CASE]):
        raise ValueError('Only the exact final incomplete opportunity may recover')
    for key in parent['case_order'][:-1]:
        row=stopped['records'][key]
        if (row['audit']['status']!='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT'
                or not row['audit']['full512']['gradient']['passes_two_percent_gate']
                or row['audit']['full512']['windows']!=512 or not row['audit']['replay']['passes']
                or row['score']['status']!='COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT'
                or 'cleanup_receipt_sha256' not in row):
            raise ValueError('Every earlier lower-rate opportunity must already be qualified and closed')


def prepare(root):
    if managed_running():raise RuntimeError('No recovery admission over a native owner')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every recovery attempt')
    prior=root/'scale-boundary-lower-rates-v1';parent_file=prior/'plan.json';result_file=prior/'report.json'
    parent=json.loads(parent_file.read_text());stopped=json.loads(result_file.read_text())
    validate_stop(root,parent,stopped)
    if stopped['plan_sha256']!=file_sha256(parent_file):raise AssertionError('Stopped registration changed')
    if (root/'scale-body-exposure-main-v1').exists():raise RuntimeError('No recovery over an admitted main')
    if shutil.disk_usage(root).free<53*1024**3:raise RuntimeError('Complete18GiB diagnostic peak plus35GiB reserve required')
    if shutil.disk_usage(root.parent/'recurrence').free<37*1024**3:raise RuntimeError('Retained ordinary prediction plus reserve required')
    inode_projection=project_snapshot_inodes(Path('.'),8)
    if os.statvfs(root).f_favail<inode_projection['required_free_inodes']:
        raise RuntimeError('Both recovery and all six future main snapshots must fit the inode reserve')
    for name,sha in parent['source_sha256'].items():
        if (name.startswith('compute_trainium/') or name in ['scripts/train_scale_registered.py',
                'scripts/train_scale_capture_invalid_gradient.py','scripts/audit_scale_registered.py',
                'scripts/score_scale_registered.py']) and file_sha256(name)!=sha:
            raise AssertionError('Qualified learning source changed: '+name)
    for name,sha in parent['numerical_gates'].items():
        if file_sha256(root/name/'completion-verification.json')!=sha:raise AssertionError('Numerical proof changed')
    for key in parent['case_order'][:-1]:
        case=parent['cases'][key];row=stopped['records'][key]
        if (file_sha256(root/case['retained_model'])!=row['score']['retained_model_sha256']
                or file_sha256(root/case['cleanup_receipt'])!=row['cleanup_receipt_sha256']):
            raise AssertionError('Completed prediction or closure changed')
    old=parent['cases'][CASE]
    if (old['peak_lr']!=.002 or old['endpoint_updates']!=300 or old['end_targets']!=157013162
            or file_sha256(root/old['initializer'])!=old['initializer_sha256']):
        raise AssertionError('Original untouched initializer and scientific opportunity required')
    old_identity=prior.name+'-'+CASE+'-warm'
    if (root/'runs'/old_identity/'progress.json').exists():raise ValueError('Inspect any recorded training before recovery')
    closed=root/'snapshots'/old_identity/'closure.json'
    if json.loads(closed.read_text())['exit_code']!=137:raise ValueError('Exact failed pre-update launch required')
    artifact=IDENTITY+'/artifacts/'+CASE
    case={**old,'artifact_directory':artifact+'/warm','replay_directory':artifact+'/replay',
        'audit_directory':IDENTITY+'/audits/'+CASE,'score_directory':IDENTITY+'/scores/'+CASE,
        'retained_model':'../recurrence/'+IDENTITY+'/retained/'+CASE+'.pt','cleanup_receipt':IDENTITY+'/cleanup/'+CASE+'.json'}
    replay={**case,'artifact_directory':case['replay_directory'],'resume':dict(directory=case['artifact_directory'],update=299),
            'save_updates':[300],'capture_before_updates':[],'audit_first_update':False}
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for name in ['scripts/run_native_isolated.sh',DOCUMENT]:sources[name]=file_sha256(name)
    worker,host=split_snapshot_sources(sources);start=time.time();names=[IDENTITY+'-'+CASE+'-'+s for s in ['warm','replay']]
    plan={**parent,'identity':IDENTITY,'registered_unix':start,'seconds_limit':7200,'absolute_deadline_unix':start+7200,
        'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host,
        'selection_parent_report':str(result_file.relative_to(root)),'selection_parent_report_sha256':file_sha256(result_file),
        'recovery_parent_plan_sha256':file_sha256(parent_file),'failed_native_closure_sha256':file_sha256(closed),
        'cases':{CASE:case,CASE+'-replay':replay},'case_order':[CASE],
        'phases':{n:dict(case=CASE if i==0 else CASE+'-replay') for i,n in enumerate(names)},
        'root_comm_ports':reserve_ports(root,names),'native_phase_seconds_limits':dict(zip(names,[1500,600])),
        'case_native_phases':{CASE:names},'allowed_tune_artifact_roots':[IDENTITY+'/artifacts'],
        'actual_new_tune_update_allowance':301,'inode_projection':inode_projection,'scope':Path(DOCUMENT).read_text()}
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:prepare(a.root)
    else:
        engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.run(a.root)
