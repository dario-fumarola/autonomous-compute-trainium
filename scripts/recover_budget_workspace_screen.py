"""One explicit restart after endpoint capture OOM; scientific recipe unchanged."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts import run_budget_workspace_screen as runner

ID='probability-budget-cue-screen-recovery-v1'
PUBLIC=Path('results/probability_workspace_20261029/budget-cue-screen-recovery-v1')


def prepare(root):
    if managed_running():raise RuntimeError('Qualification must close first')
    gate=Path('results/probability_workspace_20261029/budget-cue-capture-replay-v1/report.json')
    if json.loads(gate.read_text())['status']!='PASS_BOUNDED_CAPTURE_EXACT_REPLAY':raise AssertionError('Bounded capture qualification required')
    old=root/'probability-budget-cue-screen-v1';p=json.loads((old/'plan.json').read_text());phase=p['arms']['workspace']['phase']
    closure=root/'snapshots'/phase/'closure.json'
    if json.loads(closure.read_text())['exit_code']==0:raise AssertionError('Expected preserved failed attempt')
    progress=json.loads((root/'runs'/phase/'progress.json').read_text())
    if progress['update']!=1530 or list((root/'runs'/phase).glob('PRIVATE-u*.pt')):raise AssertionError('Exact lost endpoint state required')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Projected inode reserve')
    if shutil.disk_usage(root).free<38*1024**3 or shutil.disk_usage('/').free<35*1024**3:raise RuntimeError('Reserve plus3GiB')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/BUDGET_CONDITIONED_SCREEN_20261029.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);p['arms']['workspace'].update(phase=ID+'-workspace',recovery_checkpoint_updates=[768])
    names=[p['arms']['workspace']['phase'],ID+'-evaluation'];now=time.time()
    remaining=p['absolute_deadline_unix']-now
    if remaining<3600:raise RuntimeError('Original two-hour deadline has insufficient bounded recovery allowance')
    p.update(identity=ID,registered_unix=now,seconds_limit=remaining,checkpoint_rank_batch=4,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        capture_qualification_sha256=file_sha256(gate),original_plan_sha256=file_sha256(old/'plan.json'),
        failed_closure_sha256=file_sha256(closure),failed_updates=1530,failed_real_targets=progress['cumulative_real_targets'],
        failed_costs='results/probability_workspace_20261029/budget-cue-screen-v1/execution-costs.json',
        recovery_reason='All1530 updates completed;simultaneous64-rank CPU full-state capture triggered container OOM before any checkpoint. Exact original initialization/source/schedule repeated,not a new seed. Snapshot ranks now grouped4 and gradients released. Midpoint768 recovery-only checkpoint adds no quality readout or scientific endpoint. Original deadline retained.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    else:
        runner.ID=ID;runner.PUBLIC=PUBLIC;runner.run(a.root)
