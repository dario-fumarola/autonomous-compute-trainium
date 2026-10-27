"""Register and capture actual large-shape numerics/throughput, never science."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_scale_sharding_fixture import AllChipExecution,SCALE_END

IDENTITY='scale-956m-numerical-v1'
PUBLIC=Path('results/scale_956m_numerical_20261027')


def prepare(root):
    from compute_trainium.scale_sharded_candidate import scale_inventory
    if managed_running():raise RuntimeError('Actual-shape registration requires preceding owners terminal')
    out=root/IDENTITY;artifact=root.parent/'recurrence'/IDENTITY
    if out.exists() or artifact.exists() or PUBLIC.exists():raise FileExistsError('Preserve original registrations')
    initial=root.parent/'recurrence/scale-random-initializers-v1'
    initializer=json.loads((initial/'report.json').read_text())
    if initializer['status']!='PREPARED_SCALE_QUALIFICATION_RANDOM_INITIALIZERS':raise ValueError('CPU initializer missing')
    entry=next(r for r in initializer['records'] if r['role']=='fixed_two')
    if file_sha256(initial/entry['file'])!=entry['sha256']:raise AssertionError('Random initializer changed')
    prior=root/'scale-streamed-sharding-fixture-v1/report.json'
    if json.loads(prior.read_text())['status']!='PASS_SCALE_SHARDED_SOFTWARE_FIXTURE_AND_REPLAY':
        raise ValueError('Complete streamed numerical prerequisite required')
    if shutil.disk_usage(artifact.parent).free<115*1024**3:
        raise RuntimeError('Complete 80GiB actual-shape artifacts plus protected 35GiB reserve required')
    for path in [root,root.parent,Path('/')]:
        if shutil.disk_usage(path).free<36*1024**3:raise RuntimeError('Protected output reserve plus cache allowance')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<768*1024**3:raise RuntimeError('Full 256GiB worker and 512GiB host reserve required')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete snapshot inode allowance')
    sources={}
    for folder in ['compute_trainium','scripts','configs','third_party']:
        for file in Path(folder).rglob('*.py'):
            if file.is_file():sources[str(file)]=file_sha256(file)
    for file in ['docs/SCALE_ACTUAL_SHAPE_QUALIFICATION_20261027.md','tests/test_scale_model_state.py',
                 'tests/test_scale_sharded_candidate.py','tests/test_owned_native_all_chips.py','scripts/run_native_isolated.sh']:
        sources[file]=file_sha256(file)
    worker,host=split_snapshot_sources(sources);inventory=scale_inventory()[0]
    data_dir=root/'natural-append-6500m-v2';data=json.loads((data_dir/'packed/manifest.json').read_text())
    phases={IDENTITY+'-'+mode:dict(mode=mode) for mode in ['warm','replay']}
    plan=dict(identity=IDENTITY,purpose='scale_actual_shape_numerical_only',registered_unix=time.time(),
        model=inventory['config'],parameters=inventory['parameters'],world=64,accumulation=8,global_windows=512,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        phases=phases,root_comm_ports=reserve_ports(root,list(phases)),worker_memory_gib=256,
        seconds_limit=3600,phase_seconds_limit=1800,cpu_phase_seconds_limit=7200,
        absolute_deadline_unix=min(SCALE_END,time.time()+3600),scale_budget_deadline_unix=SCALE_END,
        host_artifact_root=str(artifact),container_artifact_root='/work/recurrence/'+IDENTITY,
        container_initializer='/work/recurrence/scale-random-initializers-v1/'+entry['file'],initializer_sha256=entry['sha256'],
        source_initializer_report_sha256=file_sha256(initial/'report.json'),prior_streamed_report_sha256=file_sha256(prior),
        data=dict(packed='natural-append-6500m-v2/packed',source_sha256=data['source_sha256'],
            manifest_sha256=file_sha256(data_dir/'packed/manifest.json'),fingerprint=data['fingerprint']),
        expected_artifact_peak_bytes=80*1024**3,inode_projection=projection,
        actual_qualification_update_allowance=102,scientific_updates=0,scientific_fit_admitted=False,
        scope=Path('docs/SCALE_ACTUAL_SHAPE_QUALIFICATION_20261027.md').read_text())
    if plan['absolute_deadline_unix']+7200>SCALE_END:raise TimeoutError('Full Native and CPU audit allowance must fit scale cap')
    out.mkdir();artifact.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_NOT_ACTIVATED',scientific_fit_admitted=False))


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution')
    result=dict(status='STOPPED_SCALE_ACTUAL_SHAPE_CAPTURE',records={},scientific_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('Actual-shape qualification interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for mode in ['warm','replay']:
            name=IDENTITY+'-'+mode
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                '--plan-sha',execution.sha,'--mode',mode,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            result['records'][mode]=execution.native(name,'scripts.qualify_scale_model_native',options,'CAPTURED_SCALE_ACTUAL_SHAPE_NUMERICAL')
            write_state(PUBLIC/'progress.json',result)
        result['status']='CAPTURED_SCALE_ACTUAL_SHAPE_CPU_AUDITS_PENDING'
    except BaseException as error:
        result.update(error=str(error),error_type=type(error).__name__);raise
    finally:
        result.update(plan_sha256=execution.sha,independent_cpu_gradient_gate=False,independent_replay_gate=False,
            scope='Capture only. Full512 independent upstream gradients, first update/moments and replay comparison remain mandatory. No scientific fit.')
        execution.close(out/'execution-costs.json');write_state(out/'report.json',result)
        write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        write_state(PUBLIC/'status.json',dict(status=result['status'],scientific_fit_admitted=False))
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    run(args.root) if args.run else prepare(args.root)
