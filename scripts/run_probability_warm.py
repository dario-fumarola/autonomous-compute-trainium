"""Freeze actual-shape qualification; no scientific reuse of disposable states."""
import argparse,json,os,shutil,time
from pathlib import Path
from dataclasses import asdict
from compute_trainium.model import ModelConfig
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution

ID='probability-workspace-warm-v1';PUBLIC=Path('results/probability_workspace_20261028/warm-v1')


def prepare(root):
    if managed_running():raise RuntimeError('Exclusive admission requires a closed owner')
    prerequisite=Path('results/probability_workspace_20261028/numerical-v1/report.json')
    if json.loads(prerequisite.read_text())['status']!='PASS_PROBABILITY_WORKSPACE_NUMERICAL_AND_REPLAY':raise AssertionError('Small complete fixture prerequisite')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<40*1024**3:
        raise RuntimeError('Protected floor plus complete5GiB actual-shape artifacts required')
    source=root/'probability-workspace-data-v1';r=json.loads((source/'report.json').read_text())
    if r['status']!='PASS_PROBABILITY_WORKSPACE_NATURAL_SOURCES' or r['reserved_group_overlap'] or r['source_group_overlap']:
        raise AssertionError('Source isolation gate')
    if any(file_sha256(source/n)!=h for n,h in r['files'].items()):raise AssertionError('Frozen source bytes changed')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/PROBABILITY_WORKSPACE_20261028.md','tests/test_probability_workspace.py'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[ID+'-'+m for m in ('warm','replay')];projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete projected inode allowance required')
    now=time.time();plan=dict(identity=ID,purpose='probability_workspace_actual_shape_qualification',registered_unix=now,
        world=64,worker_memory_gib=192,seconds_limit=3600,phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,
        absolute_deadline_unix=now+5400,clock_admission_waiver=True,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=m) for n,m in zip(names,('warm','replay'))},inode_projection=projection,
        prerequisite_sha256=file_sha256(prerequisite),data_relative='probability-workspace-data-v1',data_report_sha256=file_sha256(source/'report.json'),
        model_config=asdict(ModelConfig(dim=512,layers=16,heads=8,kv_heads=4,ffn_dim=1376,vocab_size=32000,max_seq=512)),
        seed=9419,lr=3e-4,batch=1,early_warm_mean_seconds_limit=10,warm_phase=names[0],
        cold_local_supported_steps=[0,1,2],warm_updates=100,initial_training_updates=3,replay_update=104,
        training_schedule='extra_steps=(update-1)%3 through103;104 uses2 for exact replay/mature audit',
        qualification_only=True,retained_for_science=False,scientific_fit_admitted=False,protected_evaluation_groups_scored=0,
        artifact_allowance_bytes=5*1024**3,scope='Disposable natural TRAIN software/throughput test. Cold local CPU gates at every supported depth;100 warm updates;full-global mature CPU gradient;exact separate-process update replay.No quality score or scientific learning claim.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');r=dict(status='STOPPED_PROBABILITY_WARM',records={},scientific_fit_admitted=False)
    try:
        for mode in ('warm','replay'):
            name=ID+'-'+mode;args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json',
                '--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name]),'--mode',mode]
            if mode=='replay':args+=['--parent','/work/instance-nvme/runs/'+ID+'-warm']
            status='COMPLETE_PROBABILITY_WARM_PENDING_GLOBAL_AUDIT' if mode=='warm' else 'PASS_PROBABILITY_WARM_EXACT_REPLAY'
            r['records'][mode]=e.native(name,'scripts.qualify_probability_warm',args,status)
            write_state(PUBLIC/'progress.json',r)
        path=out/'cpu-audit'
        r['records']['cpu_audit']=e.cpu(ID+'-cpu','scripts.audit_probability_warm',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(path)],path/'report.json','PASS_PROBABILITY_MATURE_GLOBAL_CPU')
        r['status']='PASS_PROBABILITY_ACTUAL_SHAPE_WARM_AND_RECOVERY'
    except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();prepare(a.root) if a.action=='prepare' else run(a.root)
