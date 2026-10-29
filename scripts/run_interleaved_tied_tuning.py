"""Bounded independent LR tuning of the qualified tied-table challenger."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import matrix_work
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_interleaved_frontier_qualification import mechanism_gate
from scripts.run_probability_screen import ScreenExecution

ID='interleaved-tied-ordinary-tuning-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/tied-ordinary-tuning-v1')


def prepare(root):
    gates=mechanism_gate()
    if managed_running():raise RuntimeError('Prior Native owner must close')
    qpath=Path('results/interleaved_workspace_20261029/tied-ordinary-qualification-v1/report.json')
    q=json.loads(qpath.read_text())
    if q['status']!='PASS_TIED_ORDINARY_ACTUAL_SHAPE_AND_REPLAY':raise AssertionError('Actual tied shape must qualify')
    if min(shutil.disk_usage('/').free,shutil.disk_usage(root).free)<35*1024**3 or shutil.disk_usage(root).free<39*1024**3:
        raise RuntimeError('Protected output reserves and4GiB artifacts')
    parent=root/'interleaved-lr-tuning-v1/plan.json';old=json.loads(parent.read_text())
    qualified=json.loads((root/'interleaved-tied-ordinary-qualification-v1/plan.json').read_text())
    for f in ('compute_trainium/tied_interleaved_ordinary.py','scripts/tied_interleaved_entry.py',
              'compute_trainium/interleaved_workspace.py','compute_trainium/model.py','scripts/train_interleaved_screen.py'):
        if file_sha256(f)!=qualified['source_sha256'][f]:raise AssertionError('Qualified execution must remain unchanged')
    c=ModelConfig(**qualified['model_config']);p=copy.deepcopy(old);arms={};points=[]
    for i,lr in enumerate(old['learning_rate_grid']):
        name=f'tied-lr{i}';a=copy.deepcopy(old['arms'][f'ordinary-lr{i}']);updates=len(a['depth_schedule'])
        a.update(family='ordinary',phase=ID+'-'+name,
            nominal_matrix_flops=updates*64*matrix_work(c,512,0,mode='ordinary',batch=p['batch'])['training_matmul_flops'])
        arms[name]=a;points.append(dict(label=name,arm=name,update=a['checkpoint_updates'][-1],variants=[dict(label='main',depth=0,mode='ordinary')]))
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_TIED_ORDINARY_TUNING_20261029.md',
              'docs/INTERLEAVED_TIED_ORDINARY_INVENTORY_20261029.md'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[a['phase'] for a in arms.values()]+[ID+'-evaluation']
    projection=project_snapshot_inodes(Path('.'),len(names))
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete queue inode reserve')
    now=time.time();p.update(identity=ID,registered_unix=now,seconds_limit=10800,absolute_deadline_unix=now+10800,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,names),
        phases={n:{} for n in names},inode_projection=projection,arms=arms,evaluation_points=points,model_config=c.__dict__,
        parameters=c.parameter_count(),untied_prediction_table=False,artifact_allowance_bytes=4*1024**3,
        mechanism_gate_sha256=gates,qualification_report_sha256=file_sha256(qpath),parent_plan_sha256=file_sha256(parent),
        qualification={'tied':dict(report_sha256=file_sha256(qpath),mean_warm_seconds=q['records']['warm']['mean_warm_seconds'])},
        scope='Additional conventional resource-allocation control, independently LR-tuned with unchanged authentic exposure. One seed,reused DEV. Tied shared vocabulary plus11 unique blocks. No automatic larger run or confirmation.')
    if c.parameter_count()!=51179008 or c.parameter_count()>51747328:raise AssertionError('Registered parameter ceiling')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    print(json.dumps(dict(status='REGISTERED_TIED_ORDINARY_TUNING',arms=len(arms),updates=updates,targets=a['real_targets'])))


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');result=dict(status='RUNNING_TIED_ORDINARY_TUNING',records={})
    try:
        for arm,a in e.plan['arms'].items():
            name=a['phase'];args=['--entry','train','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                '--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])]
            result['records'][arm]=e.native(name,'scripts.tied_interleaved_entry',args,'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json',result);dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.tied_interleaved_entry',
                ['--entry','scientific-audit','--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],
                dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS');write_state(PUBLIC/'progress.json',result)
        if len({result['records'][arm]['initial_model_digest'] for arm in e.plan['arms']})!=1:
            raise AssertionError('Same initial weights across independent LR opportunities')
        if any(result['records'][arm]['parameters']!=e.plan['parameters'] for arm in e.plan['arms']):raise AssertionError('Qualified parameter inventory')
        name=ID+'-evaluation';result['records']['evaluation']=e.native(name,'scripts.tied_interleaved_entry',
            ['--entry','evaluate','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,
             '--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        dest=out/'analysis';result['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_interleaved_tied_tuning',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_TIED_ORDINARY_TUNING_ANALYSIS')
        write_state(PUBLIC/'analysis.json',result['records']['analysis']);result['status']='COMPLETE_TIED_ORDINARY_TUNING'
    except BaseException as error:
        result.update(status='STOPPED_TIED_ORDINARY_TUNING',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
