"""Actual-shape cost and full-state qualification, one disposable mode at a time."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.model import ModelConfig
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

def paths(root,mode):
    identity=f'interleaved-{mode}-warm-v1'
    return identity,root/identity,Path('results/interleaved_workspace_20261029')/(mode+'-warm-v1')

def prepare(root,mode):
    identity,out,public=paths(root,mode)
    if managed_running():raise RuntimeError('Previous Native owner must close')
    gate=Path('results/interleaved_workspace_20261029/numerical-v1/report.json');r=json.loads(gate.read_text())
    if r['status']!='PASS_INTERLEAVED_NUMERICAL_AND_REPLAY':raise AssertionError('Small all-chip gates first')
    qualified=json.loads((root/'interleaved-workspace-numerical-v1/plan.json').read_text())
    if file_sha256('compute_trainium/interleaved_workspace.py')!=qualified['source_sha256']['compute_trainium/interleaved_workspace.py']:raise AssertionError('Qualified model changed')
    if shutil.disk_usage('/').free<35*1024**3 or shutil.disk_usage(root).free<39*1024**3:raise RuntimeError('4GiB full artifact allowance plus protected reserve')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Both snapshots plus inode reserve')
    data=root/'probability-workspace-data-v1';source=json.loads((data/'report.json').read_text())
    if source['status']!='PASS_PROBABILITY_WORKSPACE_NATURAL_SOURCES' or source['reserved_group_overlap'] or source['source_group_overlap']:raise AssertionError('Source qualification')
    if any(file_sha256(data/n)!=h for n,h in source['files'].items()):raise AssertionError('Frozen source bytes changed')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_ACTUAL_SHAPE_20261029.md','tests/test_interleaved_workspace.py','tests/test_interleaved_objective.py'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[identity+'-'+m for m in ('warm','replay')]
    c=ModelConfig(dim=512,layers=6,heads=8,kv_heads=8,ffn_dim=1376,vocab_size=32000,max_seq=512);now=time.time()
    p=dict(identity=identity,purpose='interleaved_workspace_actual_shape_qualification',registered_unix=now,
      seconds_limit=5400,absolute_deadline_unix=now+5400,phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,
      clock_admission_waiver=True,world=64,worker_memory_gib=192,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
      root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=m) for n,m in zip(names,('warm','replay'))},inode_projection=projection,
      artifact_allowance_bytes=4*1024**3,model_config=c.__dict__,untied_prediction_table=True,checkpoint_rank_batch=4,
      parameters=c.parameter_count()+c.vocab_size*c.dim,seed=9823,workspace_mode=mode,batch=2,lr=.0003,
      data_relative=data.name,data_report_sha256=file_sha256(data/'report.json'),qualification_report_sha256=file_sha256(gate),
      cold_local_supported_steps=[2],depth_schedule=[2],replay_depth=2,warm_updates=100,initial_training_updates=3,replay_update=104,
      early_warm_mean_seconds_limit=8,warm_phase=names[0],qualification_only=True,retained_for_science=False,
      scientific_fit_admitted=False,research_optimizer_updates=0,protected_evaluation_groups_scored=0,
      scope='51.75M six-layer shape, global128 source windows, final authentic-token NTP, one persistent latent and two fixed Jacobi refinements.100 disposable warm updates, independent full-global mature gradients and exact separate-process replay. Same shape/scaling for ordinary and constant-pause qualification. No heldout score, scientific fitting or automatic promotion.')
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',p);write_state(public/'plan.json',p)

def run(root,mode):
    identity,out,public=paths(root,mode);e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='RUNNING_INTERLEAVED_ACTUAL_SHAPE',records={})
    try:
        for stage in ('warm','replay'):
            name=identity+'-'+stage;args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{identity}/plan.json','--plan-sha',e.sha,'--mode',stage,'--comm-port',str(e.plan['root_comm_ports'][name])]
            if stage=='replay':args+=['--parent',f'/work/instance-nvme/runs/{identity}-warm']
            status='COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if stage=='warm' else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY'
            r['records'][stage]=e.native(name,'scripts.qualify_interleaved_warm',args,status);write_state(public/'progress.json',r)
        dest=out/'cpu-audit';r['records']['audit']=e.cpu(identity+'-audit','scripts.audit_interleaved_warm',['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        r['status']='PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY'
    except BaseException as error:r.update(status='STOPPED_INTERLEAVED_ACTUAL_SHAPE',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(public/'report.json',r);e.close(public/'execution-costs.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--mode',choices=['latent','ordinary','pause'],required=True);a.add_argument('--action',choices=['prepare','run'],required=True);p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root,p.mode)
