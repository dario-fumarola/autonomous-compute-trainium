"""One fixed-budget follow-up; reuses audited controls without refitting them."""
import argparse, copy, json, os, shutil, time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution

ID='probability-fixed-one-v1'
PUBLIC=Path('results/probability_workspace_20261029/fixed-one-v1')
PARENT='probability-workspace-evaluation-recovery-v2'

def prepare(root):
    if managed_running(): raise RuntimeError('Complete the existing comparison first')
    previous=Path('results/probability_workspace_20261028/evaluation-recovery-v2')
    done=json.loads((previous/'report.json').read_text())
    if done['status']!='COMPLETE_PROBABILITY_LEARNING_SCREEN': raise AssertionError('Completed prior evidence required')
    p=copy.deepcopy(json.loads((root/PARENT/'plan.json').read_text()))
    for name in ('scripts/train_probability_screen.py','scripts/audit_probability_screen.py','scripts/evaluate_probability_screen.py','compute_trainium/probability_workspace.py','compute_trainium/probability_workspace_data.py'):
        if file_sha256(name)!=p['source_sha256'][name]: raise AssertionError('Reuse the same qualified scientific implementation')
    for relative,receipt_key in ((p['data_relative'],'data_report_sha256'),(p['evaluation_relative'],'evaluation_report_sha256')):
        base=root/relative
        if file_sha256(base/'report.json')!=p[receipt_key]:raise AssertionError('Frozen source receipt changed')
        receipt=json.loads((base/'report.json').read_text())
        if any(file_sha256(base/name)!=h for name,h in receipt['files'].items()):raise AssertionError('Frozen source bytes changed')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']: raise RuntimeError('Both snapshots must fit protected inode reserve')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<38*1024**3: raise RuntimeError('3GiB artifacts plus protected storage reserve')
    sampled=copy.deepcopy(p['arms']['workspace']); final=sampled['checkpoint_updates'][-1]
    fixed=copy.deepcopy(sampled)
    fixed.update(phase=ID+'-workspace',depth_schedule=[1]*final,checkpoint_updates=[final],audit_depths=[1])
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import matrix_work
    fixed_work=final*64*matrix_work(ModelConfig(**p['model_config']),512,1,batch=p['batch'])['training_matmul_flops']
    if fixed_work!=sampled['nominal_matrix_flops']: raise AssertionError('Fixed-one must have exactly the sampled arm total work')
    fixed['nominal_matrix_flops']=fixed_work
    p['arms']={'workspace':fixed,'sampled':sampled,**{k:copy.deepcopy(p['arms'][k]) for k in ('ordinary_data','ordinary_work','uniform')}}
    variants=lambda k,proposal='expected':dict(label='k'+str(k),depth=k,proposal=proposal)
    p['evaluation_points']=[dict(label='fixed-final',arm='workspace',update=final,variants=[variants(1),dict(label='permuted-k1',depth=1,proposal='expected',permute=True)]),
      dict(label='sampled-final',arm='sampled',update=final,variants=[variants(1),variants(2)])]
    for arm in ('ordinary_data','ordinary_work','uniform'):
        cfg=p['arms'][arm]
        p['evaluation_points'].append(dict(label=arm,arm=arm,update=cfg['checkpoint_updates'][-1],variants=[variants(2 if arm=='uniform' else 0,cfg['proposal'])]))
    reused={}
    for arm in ('sampled','ordinary_data','ordinary_work','uniform'):
        old_arm='workspace' if arm=='sampled' else arm
        audit=root/PARENT/('audit-'+old_arm)/'report.json'
        receipt=json.loads(audit.read_text())
        if receipt['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS': raise AssertionError('Reused control must be qualified')
        for row in receipt['rows']:
            path=root/'runs'/p['arms'][arm]['phase']/f"PRIVATE-u{row['update']}.pt"
            if file_sha256(path)!=row['checkpoint_sha256']: raise AssertionError('Audited control changed')
        reused[arm]=dict(original_arm=old_arm,audit_sha256=file_sha256(audit),fit_report_sha256=file_sha256(root/'runs'/p['arms'][arm]['phase']/'report.json'))
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/PROBABILITY_FIXED_ONE_20261029.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[fixed['phase'],ID+'-evaluation'];now=time.time()
    p.update(identity=ID,registered_unix=now,seconds_limit=5400,absolute_deadline_unix=now+5400,phase_seconds_limit=3600,cpu_phase_seconds_limit=1200,
      source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
      artifact_allowance_bytes=3*1024**3,reused_arms=reused,parent_report_sha256=file_sha256(previous/'report.json'),
      parent_evaluation_identity=PARENT+'-evaluation',parent_score_sha256={f'PRIVATE-rank{rank}-scores.json':file_sha256(root/'runs'/(PARENT+'-evaluation')/f'PRIVATE-rank{rank}-scores.json') for rank in range(64)},
      scope='One fixed-one-replay fit from the identical original initialization. Same source exposure,optimizer,cooldown and exactly the sampled arm matrix budget. Only its trained one-replay exit is supported. Existing controls are reused and re-evaluated identically,not counted as new replication. No automatic scaling.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    for arm,r in reused.items():
        destination=out/('audit-'+arm);destination.mkdir()
        shutil.copyfile(root/PARENT/('audit-'+r['original_arm'])/'report.json',destination/'report.json')
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_NOT_STARTED',new_fits=1,reused_controls=list(reused),new_heldout_reads=0))

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');result=dict(status='RUNNING_FIXED_ONE_SCREEN',records={},reused_costs='Original screen and recovery ledgers retained;reuse is not a free new replication')
    try:
        for arm,r in e.plan['reused_arms'].items():
            path=out/('audit-'+arm)/'report.json'
            if file_sha256(path)!=r['audit_sha256']:raise AssertionError('Reused receipt changed')
            fit=root/'runs'/e.plan['arms'][arm]['phase']/'report.json'
            if file_sha256(fit)!=r['fit_report_sha256']:raise AssertionError('Reused fit report changed')
            result['records'][arm]=json.loads(fit.read_text())
        name=e.plan['arms']['workspace']['phase']
        result['records']['workspace']=e.native(name,'scripts.train_probability_screen',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--arm','workspace','--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS')
        if len({r['initial_model_digest'] for r in result['records'].values()})!=1:raise AssertionError('Initial weights mismatch')
        destination=out/'audit-workspace'
        result['records']['audit-workspace']=e.cpu(ID+'-audit','scripts.audit_probability_screen',['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm','workspace','--output',str(destination)],destination/'report.json','PASS_PROBABILITY_SCREEN_MATURE_AUDITS')
        write_state(PUBLIC/'progress.json',result)
        name=ID+'-evaluation'
        result['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        destination=out/'analysis'
        result['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_probability_fixed_one',['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(destination)],destination/'report.json','COMPLETE_FIXED_ONE_ANALYSIS')
        write_state(PUBLIC/'analysis.json',result['records']['analysis']);result['status']='COMPLETE_FIXED_ONE_SCREEN'
    except BaseException as error:result.update(status='STOPPED_FIXED_ONE_SCREEN',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    prepare(a.root) if a.action=='prepare' else run(a.root)
