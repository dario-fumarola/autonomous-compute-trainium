"""Qualify bounded checkpoint copies against the untouched exact replay state."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution

ID='probability-budget-cue-capture-replay-v1'
PUBLIC=Path('results/probability_workspace_20261029/budget-cue-capture-replay-v1')


def main(root):
    if managed_running():raise RuntimeError('Closed owner required')
    original=root/'probability-budget-cue-warm-v1';p=json.loads((original/'plan.json').read_text())
    gate=Path('results/probability_workspace_20261029/budget-cue-warm-v1/report.json')
    if json.loads(gate.read_text())['status']!='PASS_BUDGET_CUE_ACTUAL_SHAPE':raise AssertionError('Untouched qualified state required')
    parent=root/'runs'/p['warm_phase'];bound={n:file_sha256(parent/n) for n in ('PRIVATE-u103.pt','PRIVATE-u104.pt')}
    if shutil.disk_usage(root).free<36*1024**3:raise RuntimeError('Reserve plus1GiB fixture allowance')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Projected inode reserve')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    files['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
    worker,host=split_snapshot_sources(files);name=ID+'-replay';now=time.time()
    p.update(identity=ID,registered_unix=now,seconds_limit=1800,absolute_deadline_unix=now+1800,phase_seconds_limit=1200,
        checkpoint_rank_batch=4,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,[name]),phases={name:dict(mode='replay')},inode_projection=projection,
        artifact_allowance_bytes=1024**3,parent_state_sha256=bound,original_qualification_sha256=file_sha256(gate),
        scope='One disposable exact update104 replay from unchanged qualified update103. Capture CPU state on at most4 ranks concurrently after releasing gradients;compare every model/AdamW/RNG bit to original update104. No scientific fit,no new data/quality score,no relaxed memory/numerical criterion.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='RUNNING_BOUNDED_CAPTURE_REPLAY',records={})
    try:
        r['records']['replay']=e.native(name,'scripts.qualify_probability_warm',
            ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,
             '--mode','replay','--parent','/work/instance-nvme/runs/'+p['warm_phase'],
             '--comm-port',str(p['root_comm_ports'][name])],'PASS_PROBABILITY_WARM_EXACT_REPLAY')
        if any(file_sha256(parent/n)!=h for n,h in bound.items()):raise AssertionError('Original fixture bytes changed')
        r['status']='PASS_BOUNDED_CAPTURE_EXACT_REPLAY'
    except BaseException as error:r.update(status='STOPPED_BOUNDED_CAPTURE_REPLAY',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
