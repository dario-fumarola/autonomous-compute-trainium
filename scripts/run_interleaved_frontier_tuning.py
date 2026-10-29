"""Conditional independent LR screen for either pre-inventoried untied shape."""
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
from scripts.run_interleaved_frontier_qualification import mechanism_gate,paths,SHAPES
from scripts.run_probability_screen import ScreenExecution


def locations(root,shape):
    identity=f'interleaved-frontier-{shape}-tuning-v1'
    return identity,root/identity,Path('results/interleaved_workspace_20261029')/f'frontier-{shape}-tuning-v1'


def prepare(root,shape):
    gates=mechanism_gate()
    comparison=Path('results/interleaved_workspace_20261029/tied-comparison-v1/report.json')
    cr=json.loads(comparison.read_text())
    if cr['status']!='COMPLETE_TIED_ORDINARY_PAIRED_COMPARISON' or not cr['useful_equal_data_screen']:
        raise AssertionError('Exact sequential candidate must first survive bracketed tied challenger')
    if managed_running():raise RuntimeError('Prior Native owner must close')
    identity,out,public=locations(root,shape);qid,qout,qpublic=paths(root,shape)
    q=json.loads((qpublic/'report.json').read_text())
    if q['status']!='PASS_INTERLEAVED_FRONTIER_QUALIFICATION':raise AssertionError('Actual conventional shape must qualify')
    if min(shutil.disk_usage('/').free,shutil.disk_usage(root).free)<35*1024**3 or shutil.disk_usage(root).free<39*1024**3:
        raise RuntimeError('Protected output reserves plus4GiB artifacts')
    parent=root/'interleaved-lr-tuning-v1/plan.json';old=json.loads(parent.read_text());qualified=json.loads((qout/'plan.json').read_text())
    for f in ('compute_trainium/interleaved_model_factory.py','compute_trainium/interleaved_workspace.py',
              'compute_trainium/grouped_interleaved_workspace.py','compute_trainium/model.py','scripts/train_interleaved_screen.py'):
        if file_sha256(f)!=qualified['source_sha256'][f]:raise AssertionError('Qualified execution unchanged')
    c=ModelConfig(**qualified['model_config']);p=copy.deepcopy(old);arms={};points=[]
    for i,lr in enumerate(old['learning_rate_grid']):
        name=f'{shape}-lr{i}';a=copy.deepcopy(old['arms'][f'ordinary-lr{i}']);updates=len(a['depth_schedule'])
        a.update(family='ordinary',phase=identity+'-'+name,
            nominal_matrix_flops=updates*64*matrix_work(c,512,0,mode='ordinary',batch=p['batch'])['training_matmul_flops'])
        arms[name]=a;points.append(dict(label=name,arm=name,update=a['checkpoint_updates'][-1],variants=[dict(label='main',depth=0,mode='ordinary')]))
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_ORDINARY_FRONTIER_INVENTORY_20261029.md',
              'docs/INTERLEAVED_FRONTIER_TUNING_20261029.md'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[a['phase'] for a in arms.values()]+[identity+'-evaluation']
    projection=project_snapshot_inodes(Path('.'),len(names))
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete queue inode reserve')
    now=time.time();p.update(identity=identity,registered_unix=now,seconds_limit=10800,absolute_deadline_unix=now+10800,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,names),
        phases={n:{} for n in names},inode_projection=projection,arms=arms,evaluation_points=points,model_config=c.__dict__,
        parameters=c.parameter_count()+c.vocab_size*c.dim,untied_prediction_table=True,artifact_allowance_bytes=4*1024**3,
        mechanism_gate_sha256=gates,qualification_report_sha256=file_sha256(qpublic/'report.json'),parent_plan_sha256=file_sha256(parent),
        tied_comparison_sha256=file_sha256(comparison),
        qualification={shape:dict(report_sha256=file_sha256(qpublic/'report.json'),mean_warm_seconds=q['records']['warm']['mean_warm_seconds'])},
        scope='Pre-inventoried untied conventional shape, independently LR-tuned at unchanged authentic exposure. One seed,reused DEV; no full conventional design-space or training efficiency claim, no automatic main.')
    if p['parameters']>51747328:raise AssertionError('Candidate parameter ceiling')
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',p);write_state(public/'plan.json',p)


def run(root,shape):
    identity,out,public=locations(root,shape);e=ScreenExecution(root,out/'plan.json',out/'execution')
    result=dict(status='RUNNING_ORDINARY_FRONTIER_TUNING',records={})
    try:
        for arm,a in e.plan['arms'].items():
            name=a['phase'];args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{identity}/plan.json',
                '--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])]
            result['records'][arm]=e.native(name,'scripts.train_interleaved_screen',args,'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            write_state(public/'progress.json',result);dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(identity+'-audit-'+arm,'scripts.audit_interleaved_screen',
                ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],
                dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS');write_state(public/'progress.json',result)
        if len({result['records'][arm]['initial_model_digest'] for arm in e.plan['arms']})!=1:raise AssertionError('Common initialization across LRs')
        if any(result['records'][arm]['parameters']!=e.plan['parameters'] for arm in e.plan['arms']):raise AssertionError('Qualified parameter inventory')
        name=identity+'-evaluation';result['records']['evaluation']=e.native(name,'scripts.evaluate_interleaved_screen',
            ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{identity}/plan.json','--plan-sha',e.sha,
             '--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        dest=out/'analysis';result['records']['analysis']=e.cpu(identity+'-analysis','scripts.analyze_interleaved_frontier_tuning',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_ORDINARY_FRONTIER_TUNING_ANALYSIS')
        write_state(public/'analysis.json',result['records']['analysis']);result['status']='COMPLETE_ORDINARY_FRONTIER_TUNING'
    except BaseException as error:
        result.update(status='STOPPED_ORDINARY_FRONTIER_TUNING',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(public/'report.json',result);e.close(public/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shape',choices=SHAPES,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root,a.shape)
