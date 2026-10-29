"""Bounded small-array Native qualification, never a scientific fit."""
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

ID='interleaved-workspace-numerical-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/numerical-v1')

def prepare(root):
    if managed_running():raise RuntimeError('Previous Native owner must close')
    projection=project_snapshot_inodes(Path('.'),2)
    if min(shutil.disk_usage('/').free,shutil.disk_usage(root).free)<36*1024**3:raise RuntimeError('Protected disk floor plus fixture artifacts')
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Both snapshots plus inode reserve')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','tests/test_interleaved_workspace.py','docs/INTERLEAVED_WORKSPACE_QUALIFICATION_20261029.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[ID+'-'+m for m in ('warm','replay')]
    c=ModelConfig(dim=128,layers=2,heads=4,kv_heads=2,ffn_dim=256,vocab_size=1024,max_seq=64);now=time.time()
    p=dict(identity=ID,purpose='interleaved_workspace_software_qualification',registered_unix=now,
      seconds_limit=2400,absolute_deadline_unix=now+2400,phase_seconds_limit=1000,cpu_phase_seconds_limit=60,
      clock_admission_waiver=True,world=64,worker_memory_gib=128,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
      root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=m) for n,m in zip(names,('warm','replay'))},
      inode_projection=projection,artifact_allowance_bytes=1024**3,model_config=c.__dict__,untied_prediction_table=True,
      parameters=c.parameter_count()+c.vocab_size*c.dim,seed=9821,length=32,
      update_cases=[[0,'ordinary'],[0,'pause'],[2,'latent'],[2,'latent'],[2,'latent']],
      qualification_only=True,scientific_fit_admitted=False,retained_for_science=False,research_examples=0,research_optimizer_updates=0,
      scope='Small random software arrays only; all64 replicas, independent CPU gradients, optimizer and moments, exact separate-process next-update replay. No quality result, source text, scientific fit or automatic promotion. Unchanged2% family/update/moment and.003 loss gates.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='RUNNING_INTERLEAVED_NUMERICAL',records={})
    try:
        for mode in ('warm','replay'):
            name=ID+'-'+mode;args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,'--mode',mode,'--comm-port',str(e.plan['root_comm_ports'][name])]
            if mode=='replay':args+=['--parent',f'/work/instance-nvme/runs/{ID}-warm']
            r['records'][mode]=e.native(name,'scripts.qualify_interleaved_workspace',args,'PASS_INTERLEAVED_WORKSPACE_SOFTWARE_FIXTURE');write_state(PUBLIC/'progress.json',r)
        r['status']='PASS_INTERLEAVED_NUMERICAL_AND_REPLAY'
    except BaseException as error:r.update(status='STOPPED_INTERLEAVED_NUMERICAL',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run'],required=True);p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root)
