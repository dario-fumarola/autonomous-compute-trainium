"""Three fresh controlled fits after complete untied-interface qualification."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution
ID='probability-untied-screen-v1'
PUBLIC=Path('results/probability_workspace_20261029/untied-screen-v1')
PARENT='probability-fixed-one-v1'

def prepare(root):
    if managed_running():raise RuntimeError('Qualifications must close first')
    gate=Path('results/probability_workspace_20261029/untied-warm-v1/report.json');g=json.loads(gate.read_text())
    if g['status']!='PASS_UNTIED_PROBABILITY_ACTUAL_SHAPE':raise AssertionError('Actual-shape audit and exact replay first')
    if g['records']['warm']['mean_warm_seconds']>3:raise RuntimeError('Measured execution cost exceeds bounded screen allowance')
    qualified=json.loads((root/'probability-untied-warm-v1/plan.json').read_text())
    for name in ('compute_trainium/untied_probability_workspace.py','compute_trainium/probability_model_factory.py','compute_trainium/pretraining_optimizer.py','compute_trainium/probability_workspace.py','scripts/train_probability_screen.py','scripts/audit_probability_screen.py','scripts/evaluate_probability_screen.py'):
        if file_sha256(name)!=qualified['source_sha256'][name]:raise AssertionError('Qualified implementation changed')
    p=copy.deepcopy(json.loads((root/PARENT/'plan.json').read_text()))
    for relative,key in ((p['data_relative'],'data_report_sha256'),(p['evaluation_relative'],'evaluation_report_sha256')):
        base=root/relative
        if file_sha256(base/'report.json')!=p[key]:raise AssertionError('Frozen source receipt changed')
        receipt=json.loads((base/'report.json').read_text())
        if any(file_sha256(base/n)!=h for n,h in receipt['files'].items()):raise AssertionError('Frozen source bytes changed')
    projection=project_snapshot_inodes(Path('.'),4)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Four snapshots plus inode reserve')
    if shutil.disk_usage(root).free<42*1024**3 or shutil.disk_usage('/').free<35*1024**3:raise RuntimeError('7GiB artifacts plus protected reserve')
    arms={k:copy.deepcopy(p['arms'][k]) for k in ('workspace','ordinary_data','ordinary_work')}
    for name,arm in arms.items():
        final=arm['checkpoint_updates'][-1]
        arm.update(phase=ID+'-'+name.replace('_','-'),checkpoint_updates=[final],recovery_checkpoint_updates=[768 if final==1530 else 1792])
    p['arms']=arms
    p['evaluation_points']=[dict(label=label,arm=name,update=cfg['checkpoint_updates'][0],variants=[dict(label='k'+str(k),depth=k,proposal='expected')]+([dict(label='permuted-k1',depth=1,proposal='expected',permute=True)] if k==1 else [])) for name,label,k,cfg in [(n,l,k,arms[n]) for n,l,k in [('workspace','untied-one',1),('ordinary_data','untied-ordinary-data',0),('ordinary_work','untied-ordinary-work',0)]]]
    oldeval=root/'runs'/(PARENT+'-evaluation')
    if json.loads((oldeval/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited tied readouts required')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/UNTIED_PROBABILITY_SCREEN_20261029.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[a['phase'] for a in arms.values()]+[ID+'-evaluation'];now=time.time()
    p.update(identity=ID,registered_unix=now,seconds_limit=10800,absolute_deadline_unix=now+10800,
      phase_seconds_limit=3600,cpu_phase_seconds_limit=1800,remaining_budget_conditioning=False,untied_prediction_table=True,checkpoint_rank_batch=4,
      parameters=79184384,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
      root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
      artifact_allowance_bytes=7*1024**3,qualification_report_sha256=file_sha256(gate),reused_arms={},
      parent_evaluation_identity=PARENT+'-evaluation',parent_score_sha256={f'PRIVATE-rank{rank}-scores.json':file_sha256(oldeval/f'PRIVATE-rank{rank}-scores.json') for rank in range(64)},
      scope='Three fresh untied fits with identical initialization,parameters,source order and uniform NTP. Only fixed-one exit supported for workspace. Both input and output tables exempt from decay. Tied controls reused as historical factorial contrasts,not new replication. No budget cues,router,extra objective,larger exposure or automatic promotion. One common LR is a screen,not a tuned conventional frontier.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='RUNNING_UNTIED_PROBABILITY_SCREEN',records={},new_fits=3)
    try:
        for arm,cfg in e.plan['arms'].items():
            name=cfg['phase'];r['records'][arm]=e.native(name,'scripts.train_probability_screen',
              ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])],
              'COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json',r);dest=out/('audit-'+arm)
            r['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.audit_probability_screen',
              ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_PROBABILITY_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC/'progress.json',r)
        if len({r['records'][n]['initial_model_digest'] for n in e.plan['arms']})!=1:raise AssertionError('Initial parameters must match across every new arm')
        if any(r['records'][n]['parameters']!=e.plan['parameters'] for n in e.plan['arms']):raise AssertionError('Parameter parity')
        name=ID+'-evaluation';r['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',
          ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        dest=out/'analysis';r['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_untied_probability',
          ['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_UNTIED_PROBABILITY_ANALYSIS')
        write_state(PUBLIC/'analysis.json',r['records']['analysis']);r['status']='COMPLETE_UNTIED_PROBABILITY_SCREEN'
    except BaseException as error:r.update(status='STOPPED_UNTIED_PROBABILITY_SCREEN',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run'],required=True);p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root)
