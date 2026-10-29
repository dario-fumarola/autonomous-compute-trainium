"""Explicitly gated qualification of the tied-table eleven-layer challenger."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.model import ModelConfig
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution
from scripts.run_interleaved_frontier_qualification import mechanism_gate

ID='interleaved-tied-ordinary-qualification-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/tied-ordinary-qualification-v1')


def prepare(root):
    gates=mechanism_gate()
    if managed_running():raise RuntimeError('Prior Native owner must close')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<39*1024**3:
        raise RuntimeError('Protected reserves plus4GiB artifacts')
    parent=root/'interleaved-mixed-round-qualification-v1/plan.json';p=copy.deepcopy(json.loads(parent.read_text()))
    c=ModelConfig(dim=512,layers=11,heads=8,kv_heads=8,ffn_dim=1376,vocab_size=32000,max_seq=512)
    if c.parameter_count()!=51179008 or c.parameter_count()>51747328:raise AssertionError('Inventoried tied shape ceiling')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_TIED_ORDINARY_INVENTORY_20261029.md','tests/test_tied_interleaved_ordinary.py'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[ID+'-'+s for s in ('warm','replay')]
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete snapshot/inode reserve')
    now=time.time();p.update(identity=ID,registered_unix=now,absolute_deadline_unix=now+5400,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,model_config=c.__dict__,parameters=c.parameter_count(),
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        untied_prediction_table=False,workspace_mode='ordinary',cold_local_supported_steps=[0],depth_schedule=[0],replay_depth=0,
        warm_phase=names[0],mechanism_gate_sha256=gates,model_factory='compute_trainium.tied_interleaved_ordinary.make_model',
        worker_entry='scripts.tied_interleaved_entry explicitly substitutes the tied factory into unchanged qualified engines',
        scope='Disposable actual-shape tied ordinary qualification only; independent cold/mature gradients and exact restart. No heldout data,scientific initialization,training or automatic promotion.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution');result=dict(status='RUNNING_TIED_ORDINARY_QUALIFICATION',records={})
    try:
        for stage in ('warm','replay'):
            name=ID+'-'+stage;args=['--entry','qualification','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                '--plan-sha',e.sha,'--mode',stage,'--comm-port',str(e.plan['root_comm_ports'][name])]
            if stage=='replay':args+=['--parent',f'/work/instance-nvme/runs/{ID}-warm']
            status='COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if stage=='warm' else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY'
            result['records'][stage]=e.native(name,'scripts.tied_interleaved_entry',args,status);write_state(PUBLIC/'progress.json',result)
        dest=out/'cpu-audit';result['records']['audit']=e.cpu(ID+'-audit','scripts.tied_interleaved_entry',
            ['--entry','audit','--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        result['status']='PASS_TIED_ORDINARY_ACTUAL_SHAPE_AND_REPLAY'
    except BaseException as error:
        result.update(status='STOPPED_TIED_ORDINARY_QUALIFICATION',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',result);e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run'],required=True)
    p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root)
