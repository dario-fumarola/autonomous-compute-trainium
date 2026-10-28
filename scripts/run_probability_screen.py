"""Register and execute one finite small-model learning screen, no auto-promotion."""
import argparse,json,math,os,random,shutil,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution

ID='probability-workspace-screen-v1'
PUBLIC=Path('results/probability_workspace_20261028/screen-v1')

class ScreenExecution(AllChipExecution):
    def native(self,*args,**kwargs):
        # Closed worker allocations may take a few seconds to return. Wait
        # within the registered budget before admission;never weaken reserve.
        start=time.time()
        while True:
            self.check()
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available>=(512+self.plan['worker_memory_gib'])*1024**3:break
            if time.time()-start>180:raise RuntimeError('Host allowance unavailable after bounded180-second admission wait')
            write_state(self.out/'admission-wait.json',dict(phase=args[0],seconds=time.time()-start,available_bytes=available))
            time.sleep(5)
        return super().native(*args,**kwargs)

def prepare(root):
    import numpy as np
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import matrix_work
    from compute_trainium.probability_workspace_data import ProbabilitySources
    if managed_running():raise RuntimeError('Exclusive admission requires completed qualifications')
    gate=Path('results/probability_workspace_20261028/batch4-recovery-v1/report.json')
    if json.loads(gate.read_text())['status']!='PASS_PROBABILITY_BATCH4_RECOVERY_AND_MATURE_AUDIT':raise AssertionError('Complete actual-shape qualification required')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<45*1024**3:raise RuntimeError('35GiB reserve plus10GiB artifacts')
    data=ProbabilitySources(root/'probability-workspace-data-v1');evaluation=root/'probability-workspace-evaluation-v1'
    er=json.loads((evaluation/'report.json').read_text())
    if er['source_group_overlap'] or er['protected_group_overlap'] or er['model_scores']:raise AssertionError('Source-only evaluation preparation gate')
    for base,report in [(data.root,data.report),(evaluation,er)]:
        if any(file_sha256(base/n)!=h for n,h in report['files'].items()):raise AssertionError('Frozen data bytes changed')
    batch=4;global_windows=64*batch
    counts=np.count_nonzero(np.asarray(data.domains[1:global_windows*512*4500+1]).reshape(4500,global_windows*512),axis=1)
    cumulative=counts.cumsum();updates=int(np.searchsorted(cumulative,200_000_000))+1;updates=3*math.ceil(updates/3)
    early=3*math.ceil((int(np.searchsorted(cumulative,64_000_000))+1)/3)
    c=ModelConfig(dim=512,layers=16,heads=8,kv_heads=4,ffn_dim=1376,vocab_size=32000,max_seq=512)
    ops={k:64*matrix_work(c,512,k,batch=batch)['training_matmul_flops'] for k in (0,1,2)}
    rng=random.Random(9512);schedule=[]
    for _ in range(updates//3):
        block=[0,1,2];rng.shuffle(block);schedule+=block
    work=sum(ops[k] for k in schedule);ordinary_updates=math.ceil(work/ops[0])
    if ordinary_updates*global_windows+global_windows>=1_000_000:raise AssertionError('Audit prefix overlap')
    arms={
      'workspace':dict(proposal='expected',depth_schedule=schedule,checkpoint_updates=[early,updates],audit_depths=[0,1,2]),
      'ordinary_data':dict(proposal='expected',depth_schedule=[0]*updates,checkpoint_updates=[early,updates],audit_depths=[0]),
      'ordinary_work':dict(proposal='expected',depth_schedule=[0]*ordinary_updates,checkpoint_updates=[ordinary_updates],audit_depths=[0]),
      'uniform':dict(proposal='uniform',depth_schedule=[2]*updates,checkpoint_updates=[updates],audit_depths=[2])}
    for name,a in arms.items():
        a.update(phase=ID+'-'+name.replace('_','-'),warmup_updates=32,decay_start=math.floor(.9*len(a['depth_schedule'])),
            real_targets=int(cumulative[len(a['depth_schedule'])-1]),
            nominal_matrix_flops=sum(64*matrix_work(c,512,k,a['proposal'],batch=batch)['training_matmul_flops'] for k in a['depth_schedule']))
    variants=[dict(label='k'+str(k),depth=k,proposal='expected') for k in (0,1,2)]
    points=[dict(label='initial',arm='initial',variants=variants)]
    for label,u in [('early',early),('final',updates)]:
        points.append(dict(label='workspace-'+label,arm='workspace',update=u,variants=variants+([dict(label='permuted-k2',depth=2,proposal='expected',permute=True)] if label=='final' else [])))
        points.append(dict(label='ordinary-data-'+label,arm='ordinary_data',update=u,variants=[variants[0]]))
    points += [dict(label='ordinary-work-final',arm='ordinary_work',update=ordinary_updates,variants=[variants[0]]),
               dict(label='uniform-final',arm='uniform',update=updates,variants=[dict(label='k2',depth=2,proposal='uniform')])]
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/PROBABILITY_LEARNING_SCREEN_20261028.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[a['phase'] for a in arms.values()]+[ID+'-evaluation']
    projection=project_snapshot_inodes(Path('.'),len(names))
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('All source snapshots/artifacts must fit inode reserve')
    now=time.time();plan=dict(identity=ID,purpose='probability_workspace_learning_screen',scientific_fit_admitted=True,
        registered_unix=now,seconds_limit=14400,absolute_deadline_unix=now+14400,phase_seconds_limit=5400,cpu_phase_seconds_limit=2400,
        clock_admission_waiver=True,world=64,worker_memory_gib=192,batch=batch,seed=9511,lr=.001,weight_decay=.1,
        lr_schedule='32-update warmup,stable peak,final10% cosine to0.1peak;every final endpoint independently cooled',
        model_config=c.__dict__,parameters=c.parameter_count(),data_relative=data.root.name,evaluation_relative=evaluation.name,
        data_report_sha256=file_sha256(data.root/'report.json'),evaluation_report_sha256=file_sha256(evaluation/'report.json'),
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        audit_window_start=1_000_000,early_mean_seconds_limit=3,arms=arms,evaluation_points=points,
        evaluation_maximum_loss_error=.0002,permutation_seed=9513,bootstrap_seed=9514,bootstrap_replicates=2000,
        practical_gain_nats=.01,artifact_allowance_bytes=10*1024**3,qualification_report_sha256=file_sha256(gate),
        tuning='One common preregistered LR,not an independently tuned conventional frontier;no LR selection from this evaluation',
        scope='Single-seed early-learning mechanism screen from random initialization;no full from-scratch,allocation,reasoning or confirmation claim. Equal-work ordinary is a tested control,not a universal frontier. No automatic larger run.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_NOT_STARTED',candidate_updates=updates,ordinary_work_updates=ordinary_updates))
    print(json.dumps(dict(candidate_updates=updates,early_update=early,ordinary_work_updates=ordinary_updates,candidate_targets=int(cumulative[updates-1]),matrix_work_ratio=work/(updates*ops[0]))),flush=True)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_PROBABILITY_LEARNING_SCREEN',records={})
    try:
        for arm,a in e.plan['arms'].items():
            name=a['phase'];args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])]
            result['records'][arm]=e.native(name,'scripts.train_probability_screen',args,'COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json',result)
            dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.audit_probability_screen',[
                '--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_PROBABILITY_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC/'progress.json',result)
        name=ID+'-evaluation'
        result['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',[
            '--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        dest=out/'analysis'
        result['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_probability_screen',[
            '--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_PROBABILITY_SCREEN_ANALYSIS')
        write_state(PUBLIC/'analysis.json',result['records']['analysis']);result['status']='COMPLETE_PROBABILITY_LEARNING_SCREEN'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();prepare(a.root) if a.action=='prepare' else run(a.root)
