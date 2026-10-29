"""Finite equal-opportunity tuning after measured persistent-state qualification."""
import argparse,json,math,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='interleaved-lr-tuning-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/tuning-v1')
GRID=[.0003,.001,.003]


def updates_for_targets(data,targets,global_windows):
    lo=1;hi=data.windows//global_windows
    if data.target_count(0,hi*global_windows)<targets:raise ValueError('Source capacity')
    while lo<hi:
        middle=(lo+hi)//2
        if data.target_count(0,middle*global_windows)>=targets:hi=middle
        else:lo=middle+1
    return lo


def prepare(root):
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_workspace import matrix_work
    if managed_running():raise RuntimeError('Exclusive queue requires completed qualifications')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<47*1024**3:raise RuntimeError('Protected reserve plus12GiB artifacts')
    gates={}
    for family,folder in [('latent','latent-grouped-v1'),('pause','pause-grouped-v1'),('ordinary','ordinary-warm-v1')]:
        path=Path('results/interleaved_workspace_20261029')/folder/'report.json'
        r=json.loads(path.read_text())
        if r['status']!='PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY':raise AssertionError('All actual-shape gates required')
        gates[family]=dict(report_sha256=file_sha256(path),mean_warm_seconds=r['records']['warm']['mean_warm_seconds'])
    qualification=json.loads((root/'interleaved-latent-grouped-v1/plan.json').read_text())
    for filename in ('compute_trainium/grouped_interleaved_workspace.py','compute_trainium/interleaved_workspace.py','compute_trainium/probability_workspace.py','compute_trainium/model.py','compute_trainium/stable_target_loss.py'):
        if file_sha256(filename)!=qualification['source_sha256'][filename]:raise AssertionError('Qualified computation changed')
    data=IndexedNaturalSources(root/'indexed-natural-sources-v1');evaluation=root/'probability-workspace-evaluation-v1';er=json.loads((evaluation/'report.json').read_text())
    for base,report in [(data.root,data.report),(evaluation,er)]:
        if report['source_group_overlap'] or report['protected_group_overlap']:raise AssertionError('Source isolation')
        if any(file_sha256(base/name)!=h for name,h in report['files'].items()):raise AssertionError('Frozen source/index/evaluation bytes changed')
    source=root/data.report['source_relative']
    if any(file_sha256(source/(domain+'.u16'))!=h for domain,h in data.report['raw_token_sha256'].items()):raise AssertionError('Authentic source arrays changed')
    c=ModelConfig(**qualification['model_config']);batch=2;global_windows=128;updates=updates_for_targets(data,128_000_000,global_windows)
    audit_start=10_000_000
    if updates*global_windows>=audit_start or audit_start+global_windows>data.windows:raise AssertionError('Disjoint numerical audit range')
    arms={};points=[]
    for family in ('ordinary','latent','pause'):
        k=2 if family=='latent' else 0
        for index,lr in enumerate(GRID):
            name=family+'-lr'+str(index);phase=ID+'-'+name
            arms[name]=dict(family=family,workspace_mode=family,lr=lr,phase=phase,depth_schedule=[k]*updates,
              checkpoint_updates=[updates],audit_depths=[k],warmup_updates=64,decay_start=math.floor(.9*updates),
              real_targets=data.target_count(0,updates*global_windows),nominal_matrix_flops=updates*64*matrix_work(c,512,k,mode=family,batch=batch)['training_matmul_flops'])
            points.append(dict(label=name,arm=name,update=updates,variants=[dict(label='main',depth=k,mode=family)]))
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for name in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_LR_TUNING_20261029.md'):files[name]=file_sha256(name)
    worker,host=split_snapshot_sources(files);names=[a['phase'] for a in arms.values()]+[ID+'-evaluation'];projection=project_snapshot_inodes(Path('.'),len(names))
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Whole queue snapshot/artifact inode allowance')
    now=time.time();p=dict(identity=ID,purpose='interleaved_workspace_learning_screen',scientific_fit_admitted=True,
       registered_unix=now,seconds_limit=14400,absolute_deadline_unix=now+14400,phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,
       clock_admission_waiver=True,world=64,worker_memory_gib=192,batch=batch,seed=9901,weight_decay=.1,checkpoint_rank_batch=4,
       model_config=c.__dict__,parameters=c.parameter_count()+c.dim*c.vocab_size,untied_prediction_table=True,interleaved_implementation='grouped',
       data_relative=data.root.name,evaluation_relative=evaluation.name,data_report_sha256=file_sha256(data.root/'report.json'),evaluation_report_sha256=file_sha256(evaluation/'report.json'),
       source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
       audit_window_start=audit_start,early_mean_seconds_limit=.8,arms=arms,evaluation_points=points,evaluation_panels=['broad'],evaluation_maximum_loss_error=.0002,
       learning_rate_grid=GRID,qualification=gates,artifact_allowance_bytes=12*1024**3,automatic_promotion=False,
       scope='Equal-budget one-seed development LR selection on authentic natural tokens; fixed persistent-state computation. No fresh reserve, autonomous allocation, reasoning, training-efficiency or full from-scratch confirmation claim.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_NOT_STARTED',arms=len(arms),updates_per_arm=updates,real_targets_per_arm=data.target_count(0,updates*global_windows)))
    print(json.dumps(dict(arms=len(arms),updates_per_arm=updates,real_targets_per_arm=data.target_count(0,updates*global_windows),inode_projection=projection)))


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_INTERLEAVED_LR_TUNING',records={})
    try:
        for arm,a in e.plan['arms'].items():
            name=a['phase'];args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])]
            result['records'][arm]=e.native(name,'scripts.train_interleaved_screen',args,'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS');write_state(PUBLIC/'progress.json',result)
            dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.audit_interleaved_screen',['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS');write_state(PUBLIC/'progress.json',result)
        if len({result['records'][arm]['initial_model_digest'] for arm in e.plan['arms']})!=1:raise AssertionError('Paired initialization mismatch')
        name=ID+'-evaluation';result['records']['evaluation']=e.native(name,'scripts.evaluate_interleaved_screen',['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        dest=out/'analysis';result['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_interleaved_tuning',['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_INTERLEAVED_LR_TUNING')
        write_state(PUBLIC/'analysis.json',result['records']['analysis']);result['status']='COMPLETE_INTERLEAVED_LR_TUNING'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run'],required=True);p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root)
