"""Retain every audited fit; improve only the independent readout reference."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution
ID='probability-workspace-evaluation-recovery-v2'
PARENT='probability-workspace-screen-recovery-v1'
PUBLIC=Path('results/probability_workspace_20261028/evaluation-recovery-v2')

def prepare(root):
    if managed_running():raise RuntimeError('Prior numerical capture must be closed')
    old=root/PARENT;p=json.loads((old/'plan.json').read_text())
    diagnosis=Path('results/probability_workspace_20261029/readout-precision-v1/report.json');d=json.loads(diagnosis.read_text())
    if d['status']!='COMPLETE_NATIVE_READOUT_PRECISION_DIAGNOSIS' or not d['native_passes_original_threshold_vs_fp64'] or d['independent_fp64_agreement']>1e-10 or d['replica_maximum_error']!=0:raise AssertionError('Independent FP64 diagnosis required')
    for name in ('scripts/train_probability_screen.py','scripts/audit_probability_screen.py','compute_trainium/probability_workspace.py','compute_trainium/probability_intervention.py','scripts/analyze_probability_screen.py'):
        if file_sha256(name)!=p['source_sha256'][name]:raise AssertionError('Scientific fits,operation,intervention,analysis remain frozen')
    receipts={}
    for arm,cfg in p['arms'].items():
        path=old/('audit-'+arm)/'report.json';r=json.loads(path.read_text())
        if r['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('All fits must already pass')
        for row in r['rows']:
            if file_sha256(root/'runs'/cfg['phase']/f"PRIVATE-u{row['update']}.pt")!=row['checkpoint_sha256']:raise AssertionError('Audited checkpoint bytes changed')
        receipts[arm]=dict(audit_sha256=file_sha256(path),fit_sha256=file_sha256(root/'runs'/cfg['phase']/'report.json'))
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete remaining snapshot allowance required')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<36*1024**3:raise RuntimeError('Reserve plus1GiB evaluation allowance')
    files={str(f):file_sha256(f) for directory in ('compute_trainium','scripts','configs') for f in Path(directory).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/PROBABILITY_LEARNING_SCREEN_20261028.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);name=ID+'-evaluation';remaining=p['absolute_deadline_unix']-time.time()
    if remaining<1800:raise RuntimeError('No renewal of original four-hour cap')
    p.update(identity=ID,registered_unix=time.time(),seconds_limit=remaining,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
      root_comm_ports=reserve_ports(root,[name]),phases={name:{}},inode_projection=projection,reused_receipts=receipts,
      original_plan_sha256=file_sha256(old/'plan.json'),precision_diagnosis_sha256=file_sha256(diagnosis),
      amendment='Only independent CPU evaluation reference changes FP32 to FP64;Native predictor,weights,all readouts,panel,comparisons and2e-4 threshold unchanged. Persist each parity receipt before raising. Reuse all completed audited fits;zero training updates. Original failed readout/costs retained. Original four-hour deadline unchanged.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    for arm in p['arms']:
        destination=out/('audit-'+arm);destination.mkdir();shutil.copyfile(old/('audit-'+arm)/'report.json',destination/'report.json')

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='RUNNING_PROBABILITY_EVALUATION_RECOVERY',records={},reused_fits=True,training_updates=0)
    try:
        for arm,cfg in e.plan['arms'].items():
            fit=root/'runs'/cfg['phase']/'report.json';audit=out/('audit-'+arm)/'report.json';receipt=e.plan['reused_receipts'][arm]
            if file_sha256(fit)!=receipt['fit_sha256'] or file_sha256(audit)!=receipt['audit_sha256']:raise AssertionError('Reused receipts changed')
            r['records'][arm]=json.loads(fit.read_text());r['records']['audit-'+arm]=json.loads(audit.read_text())
        name=ID+'-evaluation'
        r['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        destination=out/'analysis'
        r['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_probability_screen',['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(destination)],destination/'report.json','COMPLETE_PROBABILITY_SCREEN_ANALYSIS')
        write_state(PUBLIC/'analysis.json',r['records']['analysis']);r['status']='COMPLETE_PROBABILITY_LEARNING_SCREEN'
    except BaseException as error:r.update(status='STOPPED_PROBABILITY_EVALUATION_RECOVERY',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args();prepare(a.root) if a.action=='prepare' else run(a.root)
