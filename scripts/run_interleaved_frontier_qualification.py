"""Gate and qualify two pre-inventoried ordinary alternatives; no science fits."""
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

SHAPES={'wider':dict(dim=576,layers=4,heads=9,kv_heads=9,ffn_dim=1384),
        'deeper':dict(dim=448,layers=10,heads=7,kv_heads=7,ffn_dim=1112)}


def paths(root,shape):
    identity='interleaved-frontier-'+shape+'-qualification-v1'
    return identity,root/identity,Path('results/interleaved_workspace_20261029')/('frontier-'+shape+'-qualification-v1')


def mechanism_gate():
    folder=Path('results/interleaved_workspace_20261029/mixed-round-tuning-v1')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_MIXED_ROUND_TUNING':
        raise AssertionError('Finish the complete mixed-round comparison first')
    analysis=json.loads((folder/'analysis.json').read_text());selected=analysis['selection']
    if not analysis['prediction_screen_pass'] or not selected['bracketed']:
        raise AssertionError('Bracketed useful prediction screen required')
    consistency=Path('results/interleaved_workspace_20261029/mixed-convergence-v1/report.json')
    cr=json.loads(consistency.read_text())
    if cr['status']!='COMPLETE_INTERLEAVED_CONVERGENCE_DIAGNOSTIC' or not cr['arms'][selected['arm']]['fixed_two_sequential_guard']:
        raise AssertionError('Selected sequential-consistency guard required')
    return {str(f):file_sha256(f) for f in (folder/'report.json',folder/'analysis.json',consistency)}


def prepare(root,shape):
    gates=mechanism_gate()
    if managed_running():raise RuntimeError('Close active Native owner first')
    identity,out,public=paths(root,shape)
    parent=root/'interleaved-mixed-round-qualification-v1/plan.json';p=copy.deepcopy(json.loads(parent.read_text()))
    q=Path('results/interleaved_workspace_20261029/mixed-round-qualification-v1/report.json')
    if json.loads(q.read_text())['status']!='PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY':raise AssertionError('Original qualified training engine required')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<39*1024**3:
        raise RuntimeError('Protected reserves plus4GiB artifact allowance')
    c=ModelConfig(**SHAPES[shape],vocab_size=32000,max_seq=512);parameters=c.parameter_count()+c.vocab_size*c.dim
    if parameters>51747328:raise AssertionError('Deployed parameter ceiling')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_FRONTIER_QUALIFICATION_20261029.md','docs/INTERLEAVED_ORDINARY_FRONTIER_INVENTORY_20261029.md'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[identity+'-'+s for s in ('warm','replay')]
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete snapshot projection')
    now=time.time();p.update(identity=identity,registered_unix=now,absolute_deadline_unix=now+5400,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,model_config=c.__dict__,parameters=parameters,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        workspace_mode='ordinary',cold_local_supported_steps=[0],depth_schedule=[0],replay_depth=0,warm_phase=names[0],
        mechanism_gate_sha256=gates,parent_qualification_report_sha256=file_sha256(q),
        scope='Disposable actual-shape numerical/runtime/replay qualification of a pre-inventoried conventional challenger. No heldout read, scientific initialization, fitting or automatic tuning/main.')
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',p);write_state(public/'plan.json',p)


def run(root,shape):
    identity,out,public=paths(root,shape);e=ScreenExecution(root,out/'plan.json',out/'execution')
    result=dict(status='RUNNING_FRONTIER_QUALIFICATION',records={})
    try:
        for stage in ('warm','replay'):
            name=identity+'-'+stage;args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{identity}/plan.json',
                '--plan-sha',e.sha,'--mode',stage,'--comm-port',str(e.plan['root_comm_ports'][name])]
            if stage=='replay':args+=['--parent',f'/work/instance-nvme/runs/{identity}-warm']
            status='COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if stage=='warm' else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY'
            result['records'][stage]=e.native(name,'scripts.qualify_interleaved_grouped',args,status);write_state(public/'progress.json',result)
        dest=out/'cpu-audit';result['records']['audit']=e.cpu(identity+'-audit','scripts.audit_interleaved_warm',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        result['status']='PASS_INTERLEAVED_FRONTIER_QUALIFICATION'
    except BaseException as error:
        result.update(status='STOPPED_FRONTIER_QUALIFICATION',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(public/'report.json',result);e.close(public/'execution-costs.json')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--shape',choices=SHAPES,required=True)
    a.add_argument('--action',choices=['prepare','run'],required=True);p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root,p.shape)
