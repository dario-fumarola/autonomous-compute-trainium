"""One new optimization fit with audited byte-pinned ordinary/final-only controls."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution
ID='probability-joint-exit-screen-v1'
PUBLIC=Path('results/probability_workspace_20261029/joint-exit-screen-v1')
PARENT='probability-untied-screen-v1'

def prepare(root):
    if managed_running():raise RuntimeError('Qualifications must finish first')
    gate=Path('results/probability_workspace_20261029/joint-exit-warm-v1/report.json');g=json.loads(gate.read_text())
    if g['status']!='PASS_JOINT_EXIT_PROBABILITY_ACTUAL_SHAPE':raise AssertionError('Complete actual-shape numerical and replay gate')
    if g['records']['warm']['mean_warm_seconds']>3:raise RuntimeError('Bounded measured throughput')
    qualified=json.loads((root/'probability-joint-exit-warm-v1/plan.json').read_text())
    for name in ('compute_trainium/probability_objective.py','compute_trainium/untied_probability_workspace.py','compute_trainium/probability_model_factory.py','compute_trainium/pretraining_optimizer.py','scripts/train_probability_screen.py','scripts/audit_probability_screen.py','scripts/evaluate_probability_screen.py'):
        if file_sha256(name)!=qualified['source_sha256'][name]:raise AssertionError('Qualified implementation changed')
    p=copy.deepcopy(json.loads((root/PARENT/'plan.json').read_text()))
    result_path=Path('results/probability_workspace_20261029/untied-screen-v1/report.json');parent=json.loads(result_path.read_text())
    if parent['status']!='COMPLETE_UNTIED_PROBABILITY_SCREEN':raise AssertionError('Completed audited controls required')
    retained={}
    for name in ('workspace','ordinary_data','ordinary_work'):
        cfg=p['arms'][name];checkpoint=root/'runs'/cfg['phase']/f"PRIVATE-u{cfg['checkpoint_updates'][-1]}.pt"
        audit=parent['records']['audit-'+name]
        if audit['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS' or {r['checkpoint_sha256'] for r in audit['rows']}!={file_sha256(checkpoint)}:raise AssertionError('Historical audited endpoint changed')
        retained[name]=dict(checkpoint_relative=str(checkpoint.relative_to(root)),sha256=file_sha256(checkpoint),record=parent['records'][name])
    for relative,key in ((p['data_relative'],'data_report_sha256'),(p['evaluation_relative'],'evaluation_report_sha256')):
        base=root/relative
        if file_sha256(base/'report.json')!=p[key]:raise AssertionError('Frozen source receipt changed')
        receipt=json.loads((base/'report.json').read_text())
        if any(file_sha256(base/n)!=h for n,h in receipt['files'].items()):raise AssertionError('Frozen source bytes changed')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Both snapshots plus inode reserve')
    if shutil.disk_usage(root).free<39*1024**3 or shutil.disk_usage('/').free<35*1024**3:raise RuntimeError('4GiB artifact allowance plus reserve')
    arm=copy.deepcopy(p['arms']['workspace'])
    arm.update(phase=ID+'-workspace',objective='uniform_exits',audit_depths=[1])
    p['arms']={'workspace':arm}
    p['evaluation_points']=[dict(label='joint',arm='workspace',update=1530,variants=[dict(label='k0',depth=0,proposal='expected'),dict(label='k1',depth=1,proposal='expected'),dict(label='permuted-k1',depth=1,proposal='expected',permute=True)])]
    oldeval=root/'runs'/(PARENT+'-evaluation')
    if json.loads((oldeval/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Parent numerical readouts required')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/JOINT_EXIT_PROBABILITY_SCREEN_20261029.md','tests/test_probability_objective.py','tests/test_joint_exit_analysis.py'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[arm['phase'],ID+'-evaluation'];now=time.time()
    p.update(identity=ID,registered_unix=now,seconds_limit=5400,absolute_deadline_unix=now+5400,
      phase_seconds_limit=2400,cpu_phase_seconds_limit=1800,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
      root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
      artifact_allowance_bytes=4*1024**3,qualification_report_sha256=file_sha256(gate),reused_arms=retained,
      parent_result_sha256=file_sha256(result_path),parent_evaluation_identity=PARENT+'-evaluation',
      parent_score_sha256={f'PRIVATE-rank{rank}-scores.json':file_sha256(oldeval/f'PRIVATE-rank{rank}-scores.json') for rank in range(64)},
      supported_exit_policy='Both0and1 supported by equal uniform per-target loss;no2exit.',
      resource_matching='Dispatched forward and backward matrix products unchanged;extra CE elementwise/runtime measured separately. Historical ordinary controls retained unchanged,not rerun seeds.',
      scope='One fresh joint-exit fit at unchanged200M targets,initialization,LR,source order and complete cooldown. Final-only and ordinary controls byte-pinned with audits and all original costs retained. No teacher,selected-target emphasis,router,new parameters,larger exposure or automatic promotion. A conventional intermediate-supervision control and fresh replication remain required for stronger positive attribution.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='RUNNING_JOINT_EXIT_SCREEN',records={},new_fits=1)
    try:
        arm='workspace';cfg=e.plan['arms'][arm];name=cfg['phase']
        r['records'][arm]=e.native(name,'scripts.train_probability_screen',
          ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS')
        write_state(PUBLIC/'progress.json',r)
        if r['records'][arm]['initial_model_digest']!=e.plan['reused_arms']['workspace']['record']['initial_model_digest']:raise AssertionError('Same initial parameters required')
        if r['records'][arm]['nominal_training_matrix_flops']!=e.plan['reused_arms']['workspace']['record']['nominal_training_matrix_flops']:raise AssertionError('Exact final-only matrix-work match')
        dest=out/'audit-workspace'
        r['records']['audit-workspace']=e.cpu(ID+'-audit-workspace','scripts.audit_probability_screen',
          ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_PROBABILITY_SCREEN_MATURE_AUDITS')
        write_state(PUBLIC/'progress.json',r)
        name=ID+'-evaluation';r['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',
          ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        dest=out/'analysis';r['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_joint_exit_probability',
          ['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_JOINT_EXIT_PROBABILITY_ANALYSIS')
        write_state(PUBLIC/'analysis.json',r['records']['analysis']);r['status']='COMPLETE_JOINT_EXIT_SCREEN'
    except BaseException as error:r.update(status='STOPPED_JOINT_EXIT_SCREEN',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run'],required=True);p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root)
