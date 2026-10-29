"""Bounded zero-update diagnosis; no fitting or scientific readout selection."""
import argparse,json,time,os
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from scripts.run_probability_screen import ScreenExecution
ID='probability-readout-precision-native-v1'
PUBLIC=Path('results/probability_workspace_20261029/readout-precision-v1')

def main(root):
    parent=json.loads((root/'probability-workspace-screen-recovery-v1/plan.json').read_text());out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('One diagnosis snapshot must fit inode reserve')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')};files['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(files)
    checkpoint='runs/'+parent['arms']['workspace']['phase']+'/PRIVATE-u1530.pt';name=ID+'-capture';now=time.time()
    p=dict(identity=ID,purpose='zero_update_readout_precision_diagnosis',registered_unix=now,seconds_limit=900,phase_seconds_limit=900,cpu_phase_seconds_limit=60,clock_admission_waiver=True,absolute_deadline_unix=now+900,worker_memory_gib=192,world=64,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[name]),phases={name:{}},inode_projection=projection,model_config=parent['model_config'],checkpoint_relative=checkpoint,checkpoint_sha256=file_sha256(root/checkpoint),permutation_seed=parent['permutation_seed'],evaluation_relative=parent['evaluation_relative'],scientific_fit_admitted=False,scope='Same frozen failed batch/checkpoint;compare Native FP32 to independent CPU FP32 and two FP64 paths. No change to2e-4 criterion,no updates,quality aggregation or model selection.')
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p);e=ScreenExecution(root,out/'plan.json',out/'execution');r=dict(status='STOPPED_READOUT_PRECISION_DIAGNOSIS')
    try:r=e.native(name,'scripts.qualify_probability_readout_precision',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][name])],'COMPLETE_NATIVE_READOUT_PRECISION_DIAGNOSIS')
    except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
