"""Register eight bounded operator diagnostics across all 64 existing ranks."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='interleaved-block-probe-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/block-probe-v1')

def prepare(root):
    if managed_running():raise RuntimeError('Previous accelerator owner must close')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<36*1024**3:raise RuntimeError('Protected reserve plus complete artifact allowance')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete inode allowance')
    files={str(f):file_sha256(f) for folder in ('compute_trainium','scripts','configs') for f in Path(folder).rglob('*.py')}
    for name in ('scripts/run_native_isolated.sh','tests/test_block_execution_probe.py'):files[name]=file_sha256(name)
    worker,host=split_snapshot_sources(files);now=time.time()
    p=dict(identity=ID,purpose='block_execution_software_probe',registered_unix=now,
           seconds_limit=1200,absolute_deadline_unix=now+1200,phase_seconds_limit=900,
           clock_admission_waiver=True,world=64,worker_memory_gib=128,
           source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
           artifact_allowance_bytes=1024**3,inode_projection=projection,
           root_comm_ports=reserve_ports(root,[ID]),phases={ID:{}},
           research_examples=0,scientific_fit_admitted=False,
           scope='Eight independent numerical cases, eight ranks each: full blocks512/1024, rotary512/1024, FFN1024,RMS1024,interleave1024,strided head512. Three post-cold warmups plus20 timed forward/backwards. Independent CPU SDPA gradients. No optimizer,real text,quality score or automatic graph adoption.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='STOPPED_BLOCK_EXECUTION_PROBE')
    try:
        r=e.native(ID,'scripts.probe_block_execution',['--root','/work/instance-nvme',
            '--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,
            '--comm-port',str(e.plan['root_comm_ports'][ID])],'COMPLETE_BLOCK_EXECUTION_PROBE')
    except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
