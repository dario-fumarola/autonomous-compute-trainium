"""One bounded zero-update readout after the complete parent comparison."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution
ID='probability-structural-readout-v1';PARENT='probability-untied-screen-v1'
PUBLIC=Path('results/probability_workspace_20261029/structural-readout-v1')

def prepare(root):
    if managed_running():raise RuntimeError('Complete existing comparison first')
    parent_public=Path('results/probability_workspace_20261029/untied-screen-v1/report.json');complete=json.loads(parent_public.read_text())
    if complete['status']!='COMPLETE_UNTIED_PROBABILITY_SCREEN':raise AssertionError('All parent fits,audits and broad comparisons first')
    parent=json.loads((root/PARENT/'plan.json').read_text());panel=root/'probability-structural-panel-v1';receipt=json.loads((panel/'report.json').read_text());rules=json.loads((PUBLIC/'prospective-rules.json').read_text())
    if receipt['status']!='FROZEN_STRUCTURAL_DEPENDENCY_PANEL' or receipt['paired_source_groups']<rules['minimum_source_groups']:raise AssertionError('Frozen adequate source yield first')
    if any(file_sha256(panel/f)!=h for f,h in receipt['files'].items()):raise AssertionError('Frozen source bytes changed')
    points=[]
    for name,depth in [('workspace',1),('ordinary_data',0),('ordinary_work',0)]:
        cfg=parent['arms'][name];audit=json.loads((root/PARENT/('audit-'+name)/'report.json').read_text())
        if audit['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('Mature audit first')
        update=cfg['checkpoint_updates'][-1];path=root/'runs'/cfg['phase']/f'PRIVATE-u{update}.pt';sha=file_sha256(path)
        if {row['checkpoint_sha256'] for row in audit['rows'] if row['update']==update}!={sha}:raise AssertionError('Audited model bytes changed')
        points.append(dict(label=name,depth=depth,phase=cfg['phase'],update=update,checkpoint_sha256=sha))
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Snapshot plus inode reserve')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<36*1024**3:raise RuntimeError('Artifact plus35GiB reserve')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')};files['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(files)
    now=time.time();name=ID+'-evaluation';p={k:parent[k] for k in ('model_config','seed','batch','world','worker_memory_gib','clock_admission_waiver')}
    p.update(identity=ID,registered_unix=now,absolute_deadline_unix=now+1800,seconds_limit=1800,phase_seconds_limit=1500,cpu_phase_seconds_limit=300,untied_prediction_table=True,remaining_budget_conditioning=False,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[name]),phases={name:{}},inode_projection=projection,readout_phase=name,structural_points=points,panel_relative=panel.name,panel_report_sha256=file_sha256(panel/'report.json'),prospective_rules_sha256=file_sha256(PUBLIC/'prospective-rules.json'),parent_report_sha256=file_sha256(parent_public),evaluation_maximum_loss_error=2e-4,bootstrap_replicates=rules['bootstrap_replicates'],bootstrap_seed=rules['bootstrap_seed'],practical_gain_nats=rules['practical_gain_nats'],source_groups=receipt['paired_source_groups'],new_fits=0,scope=rules['scope'])
    out=root/ID;out.mkdir();write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');p=e.plan;r=dict(status='RUNNING_STRUCTURAL_PROBABILITY_READOUT',records={})
    try:
        name=p['readout_phase'];r['records']['readout']=e.native(name,'scripts.evaluate_structural_probability',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][name])],'COMPLETE_STRUCTURAL_PROBABILITY_READOUT')
        dest=out/'analysis';r['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_structural_probability',['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_STRUCTURAL_PROBABILITY_ANALYSIS')
        write_state(PUBLIC/'analysis.json',r['records']['analysis']);r['status']='COMPLETE_STRUCTURAL_PROBABILITY_COMPARISON'
    except BaseException as error:r.update(status='STOPPED_STRUCTURAL_PROBABILITY_COMPARISON',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
