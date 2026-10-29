"""One all-chip timing investigation, retaining the qualified graph and states."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='interleaved-stage-profile-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/stage-profile-v1')

def prepare(root):
    if managed_running():raise RuntimeError('Previous accelerator owner must close')
    gate=Path('results/interleaved_workspace_20261029/latent-warm-v1/report.json')
    if json.loads(gate.read_text())['status']!='PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY':raise AssertionError('Complete actual-shape gate required')
    previous=json.loads((root/'interleaved-latent-warm-v1/plan.json').read_text())
    for name in ('compute_trainium/interleaved_workspace.py','compute_trainium/interleaved_objective.py'):
        if file_sha256(name)!=previous['source_sha256'][name]:raise AssertionError('Qualified graph changed')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<36*1024**3:raise RuntimeError('Complete artifact allowance plus protected reserve')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Snapshot inode allowance')
    files={str(f):file_sha256(f) for folder in ('compute_trainium','scripts','configs') for f in Path(folder).rglob('*.py')}
    files['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
    worker,host=split_snapshot_sources(files);now=time.time()
    plan=dict(identity=ID,purpose='interleaved_stage_profile',registered_unix=now,
              seconds_limit=1200,absolute_deadline_unix=now+1200,phase_seconds_limit=900,
              clock_admission_waiver=True,world=64,worker_memory_gib=192,
              artifact_allowance_bytes=1024**3,source_sha256=files,
              worker_source_sha256=worker,host_only_source_sha256=host,
              model_config=previous['model_config'],batch=2,lr=previous['lr'],
              data_relative=previous['data_relative'],untied_prediction_table=True,
              checkpoint_relative='runs/interleaved-latent-warm-v1-warm',
              checkpoint_sha256={name:file_sha256(root/'runs/interleaved-latent-warm-v1-warm'/name)
                                  for name in ('PRIVATE-u103.pt','PRIVATE-u104.pt')},
              root_comm_ports=reserve_ports(root,[ID]),phases={ID:{}},
              qualification_sha256=file_sha256(gate),scientific_fit_admitted=False,
              scope='Exactly replay qualified disposable update104, then time ten disposable updates with explicit stage synchronization. No heldout scores or scientific states.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)

def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution')
    result=dict(status='STOPPED_INTERLEAVED_STAGE_PROFILE')
    try:
        result=e.native(ID,'scripts.profile_interleaved_steps',[
            '--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
            '--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][ID])],
            'COMPLETE_INTERLEAVED_STAGE_PROFILE')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else run)(a.root)
