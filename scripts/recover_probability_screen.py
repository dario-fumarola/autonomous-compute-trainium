"""Preserve completed learning; correct only CPU state container serialization."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution

ID='probability-workspace-screen-recovery-v1'
PUBLIC=Path('results/probability_workspace_20261028/screen-recovery-v1')

def prepare(root):
    if managed_running():raise RuntimeError('Closed stopped controller required')
    old=root/'probability-workspace-screen-v1';p=json.loads((old/'plan.json').read_text())
    diagnosis=Path('results/probability_workspace_20261028/state-representation-diagnosis-v1/report.json')
    d=json.loads(diagnosis.read_text())
    if d['status']!='PASS_EXACT_CANONICAL_STATE_RESTORE' or not all(r['exact'] for r in d['rows']):raise AssertionError('Exact state diagnosis required')
    for name in ('scripts/train_probability_screen.py','compute_trainium/probability_workspace.py','compute_trainium/probability_workspace_data.py'):
        if file_sha256(name)!=p['source_sha256'][name]:raise AssertionError('Scientific training implementation must remain unchanged')
    parent=root/'runs'/p['arms']['workspace']['phase']
    if any(file_sha256(parent/f"PRIVATE-u{r['update']}.pt")!=r['checkpoint_sha256'] for r in d['rows']):raise AssertionError('Completed states changed')
    projection=project_snapshot_inodes(Path('.'),4)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Remaining inode allowance')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<41*1024**3:raise RuntimeError('Remaining6GiB artifact allowance plus reserve')
    remaining=p['absolute_deadline_unix']-time.time()
    if remaining<7200:raise RuntimeError('No renewal of original four-hour cap')
    for arm in ('ordinary_data','ordinary_work','uniform'):p['arms'][arm]['phase']=ID+'-'+arm.replace('_','-')
    names=[p['arms'][arm]['phase'] for arm in ('ordinary_data','ordinary_work','uniform')]+[ID+'-evaluation']
    sources={str(f):file_sha256(f) for directory in ('compute_trainium','scripts','configs') for f in Path(directory).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/PROBABILITY_LEARNING_SCREEN_20261028.md'):sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources)
    p.update(identity=ID,registered_unix=time.time(),seconds_limit=remaining,source_sha256=sources,worker_source_sha256=worker,
        host_only_source_sha256=host,root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        original_plan_sha256=file_sha256(old/'plan.json'),state_diagnosis_sha256=file_sha256(diagnosis),
        amendment='Only CPU restore digest now uses the same existing cpu_tree serialization as Native. Reuse the completed workspace fit without updates;unchanged data,initialization,optimizer,depth schedule,checkpoint endpoints,cooldowns,budgets,evaluation and numerical thresholds. Retain original failed CPU attempt and all original costs.',
        completed_fit_report_sha256=file_sha256(parent/'report.json'))
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_PROBABILITY_SCREEN_RECOVERY',records={},original_costs='screen-v1/execution-costs.json')
    try:
        for arm,a in e.plan['arms'].items():
            if arm=='workspace':
                path=root/'runs'/a['phase']/'report.json'
                if file_sha256(path)!=e.plan['completed_fit_report_sha256']:raise AssertionError('Completed fit receipt changed')
                result['records'][arm]=json.loads(path.read_text())
            else:
                name=a['phase'];result['records'][arm]=e.native(name,'scripts.train_probability_screen',[
                    '--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS')
            dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.audit_probability_screen',[
                '--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_PROBABILITY_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC/'progress.json',result)
        hashes={result['records'][arm]['initial_model_digest'] for arm in e.plan['arms']}
        if len(hashes)!=1:raise AssertionError('Arms did not start at identical weights')
        name=ID+'-evaluation';result['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',[
            '--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        dest=out/'analysis';result['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_probability_screen',[
            '--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_PROBABILITY_SCREEN_ANALYSIS')
        write_state(PUBLIC/'analysis.json',result['records']['analysis']);result['status']='COMPLETE_PROBABILITY_LEARNING_SCREEN'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    prepare(a.root) if a.action=='prepare' else run(a.root)
