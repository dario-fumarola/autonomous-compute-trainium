"""Three independently tuned arms; equal nominal work, never automatic admission."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from compute_trainium.chunked_persistent_factory import matrix_work
from compute_trainium.execution_isolation import managed_running
from compute_trainium.model import ModelConfig
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='chunked-persistent-screen-v1'
PUBLIC=Path('results/chunked_persistent_workspace_20261029/learning-v1')
GRID=[.0003,.001,.003]
FAMILIES={'ordinary':1,'generated':2,'constant':3}


def build_arms(old,config,data):
    batch=old['batch'];world=old['world'];updates=1957
    target_work=updates*world*matrix_work(config,512,2,batch=batch)['training_matmul_flops']
    arms,points={},[]
    for family,graph in FAMILIES.items():
        per_update=world*matrix_work(config,512,graph,batch=batch)['training_matmul_flops']
        count=(target_work+per_update-1)//per_update
        if family!='ordinary' and count!=updates:
            raise AssertionError('Generated and constant operations must match')
        if not target_work<=count*per_update<target_work+per_update:
            raise AssertionError('Never underfund the ordinary work control')
        for i,lr in enumerate(GRID):
            name=f'{family}-lr{i}'
            a=copy.deepcopy(old['arms'][f'ordinary-lr{i}'])
            if a['lr']!=lr or a['checkpoint_updates']!=[updates]:
                raise AssertionError('Original full independent LR opportunities required')
            a.update(family=family,phase=ID+'-'+name,workspace_mode='registered',
                graph_id=graph,depth_schedule=[graph]*count,audit_depths=[graph],
                checkpoint_updates=[count],decay_start=math.floor(.9*count),
                real_targets=data.target_count(0,count*world*batch),
                nominal_matrix_flops=count*per_update)
            arms[name]=a
            points.append(dict(label=name,arm=name,update=count,
                variants=[dict(label='main',depth=graph,mode='registered')]))
    return arms,points


def admission_cost(arms,warm):
    if set(warm)!={'0','1','2','3'} or any(not 0<float(x)<=1 for x in warm.values()):
        raise AssertionError('Complete bounded measured graph times required')
    measured=sum(len(a['depth_schedule'])*warm[str(a['graph_id'])] for a in arms.values())
    projected=1.5*measured+240*len(arms)+900
    if projected>14400:
        raise RuntimeError('Complete tuned screen must fit the four-hour ceiling with measured headroom')
    return dict(measured_fit_seconds=measured,headroom_projected_seconds=projected,
                fit_factor=1.5,per_fit_other_seconds=240,evaluation_readout_seconds=900)


def prepare(root):
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources
    qpath=Path('results/chunked_persistent_workspace_20261029/numerical-v1/report.json')
    q=json.loads(qpath.read_text())
    if q['status']!='PASS_CHUNKED_PERSISTENT_ACTUAL_SHAPE':
        raise AssertionError('Actual shape, full gradients, exact replay and runtime must qualify first')
    if managed_running():raise RuntimeError('Previous Native owner must close')
    if shutil.disk_usage('/').free<35*1024**3 or shutil.disk_usage(root).free<44*1024**3:
        raise RuntimeError('Nine GiB artifact allowance above protected output reserves')
    qp=root/'chunked-persistent-qualification-v1/plan.json';qualified=json.loads(qp.read_text())
    for f in ('compute_trainium/chunked_persistent_workspace.py','compute_trainium/chunked_persistent_factory.py',
              'compute_trainium/depth_causal_workspace.py','compute_trainium/tied_interleaved_ordinary.py',
              'compute_trainium/model.py','scripts/train_interleaved_screen.py',
              'scripts/audit_interleaved_screen.py','scripts/evaluate_interleaved_screen.py'):
        if file_sha256(f)!=qualified['source_sha256'][f]:raise AssertionError('Qualified execution source changed')
    oldpath=root/'interleaved-lr-tuning-v1/plan.json';old=json.loads(oldpath.read_text())
    data=IndexedNaturalSources(root/old['data_relative']);evaluation=root/old['evaluation_relative']
    er=json.loads((evaluation/'report.json').read_text())
    for base,report in [(data.root,data.report),(evaluation,er)]:
        if report['source_group_overlap'] or report['protected_group_overlap']:
            raise AssertionError('Source separation must remain intact')
        if any(file_sha256(base/f)!=h for f,h in report['files'].items()):
            raise AssertionError('Frozen source/evaluation bytes changed')
    source=root/data.report['source_relative']
    if any(file_sha256(source/(d+'.u16'))!=h for d,h in data.report['raw_token_sha256'].items()):
        raise AssertionError('Original authentic corpus changed')
    c=ModelConfig(**qualified['model_config']);arms,points=build_arms(old,c,data)
    costs=admission_cost(arms,q['records']['warm']['warm_by_depth'])
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/CHUNKED_PERSISTENT_LEARNING_20261029.md',
              'tests/test_chunked_persistent_screen.py'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[a['phase'] for a in arms.values()]+[ID+'-evaluation']
    projection=project_snapshot_inodes(Path('.'),len(names))
    if os.statvfs(root).f_favail<projection['required_free_inodes']:
        raise RuntimeError('All ten snapshots must fit above protected inode reserve')
    now=time.time();p=copy.deepcopy(old)
    p.update(identity=ID,registered_unix=now,seconds_limit=14400,absolute_deadline_unix=now+14400,
        phase_seconds_limit=3600,cpu_phase_seconds_limit=1200,early_mean_seconds_limit=1.2,
        arms=arms,evaluation_points=points,evaluation_files=er['files'],model_config=c.__dict__,
        parameters=c.parameter_count(),untied_prediction_table=False,source_stride=128,
        workspace_mechanism='chunked_persistent_generated_state',worker_entry='scripts.chunked_persistent_scientific_entry',
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        artifact_allowance_bytes=9*1024**3,measured_budget=costs,
        qualification_report_sha256=file_sha256(qpath),qualification_plan_sha256=file_sha256(qp),
        prior_plan_sha256=file_sha256(oldpath),practical_gain_nats=.01,
        bootstrap_seed=40513,bootstrap_replicates=2000,exact_readout_workers=4,
        exact_readout_threads=8,exact_readout_seconds_limit=1200,
        protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
        scope='Single-seed independently tuned matched nominal matrix-work discovery on the strongest tied ordinary body. Generated and constant persistent slots every128 source positions; ordinary receives at least the same matrix work using equally chunked execution. Uniform source-target NTP and original paired random initialization. Reused DEV; no hardware-time efficiency, adaptive allocation, extra-budget scaling, general reasoning or confirmation claim.')
    if c.parameter_count()!=51179008:raise AssertionError('Strongest tied body required')
    if max(len(a['depth_schedule']) for a in arms.values())*128>=p['audit_window_start']:
        raise AssertionError('Numerical TRAIN audit windows stay unfitted')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution')
    result=dict(status='RUNNING_CHUNKED_PERSISTENT_SCREEN',records={})
    def stop(number,frame):raise KeyboardInterrupt('Sparse screen owner interrupted')
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for arm,a in e.plan['arms'].items():
            result['records'][arm]=e.native(a['phase'],'scripts.chunked_persistent_scientific_entry',
                ['--entry','train','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                 '--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][a['phase']])],
                'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json',result);dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.chunked_persistent_scientific_entry',
                ['--entry','scientific-audit','--root',str(root),'--plan',str(out/'plan.json'),
                 '--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC/'progress.json',result)
        fits=[result['records'][arm] for arm in e.plan['arms']]
        if len({r['initial_model_digest'] for r in fits})!=1 or any(r['parameters']!=e.plan['parameters'] for r in fits):
            raise AssertionError('Exact same paired initializer and parameter inventory required')
        old=json.loads(Path('results/interleaved_workspace_20261029/tied-ordinary-tuning-v1/report.json').read_text())
        if fits[0]['initial_model_digest']!=old['records']['tied-lr0']['initial_model_digest']:
            raise AssertionError('Original strong-ordinary initializer must match')
        name=ID+'-evaluation'
        result['records']['evaluation']=e.native(name,'scripts.chunked_persistent_scientific_entry',
            ['--entry','evaluate','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
             '--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],
            'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        for action,status in [('select','COMPLETE_CHUNKED_PERSISTENT_SELECTION'),
                              ('readout','COMPLETE_CHUNKED_PERSISTENT_EXACT_READOUT'),
                              ('compare','COMPLETE_CHUNKED_PERSISTENT_ANALYSIS')]:
            dest=out/action
            module='scripts.readout_chunked_persistent' if action=='readout' else 'scripts.analyze_chunked_persistent'
            args=['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)]
            if action!='readout':args+=['--action',action]
            result['records'][action]=e.cpu(ID+'-'+action,module,args,dest/'report.json',status)
            write_state(PUBLIC/(action+'.json'),result['records'][action])
        result['status']='COMPLETE_CHUNKED_PERSISTENT_SCREEN'
    except BaseException as error:
        result.update(status='STOPPED_CHUNKED_PERSISTENT_SCREEN',error_type=type(error).__name__,error=str(error));raise
    finally:
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        e.close(PUBLIC/'execution-costs.json')
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else run)(a.root)
