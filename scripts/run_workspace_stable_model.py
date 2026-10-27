"""Fixed-point actual-size stable-loss requalification;no updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
IDENTITY='workspace-stable-model-fixed-points-v1'
PUBLIC=Path('results/workspace_stable_model_20261027')


def prepare(root):
    if managed_running():raise RuntimeError('Exclusive fixed-point qualification')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every fixed-point attempt')
    reference=root/'workspace-stable-references-v1/report.json';ref=json.loads(reference.read_text())
    fixture=root/'workspace-stable-loss-fixture-v1/report.json'
    if ref['status']!='PASS_WORKSPACE_STABLE_CPU_REFERENCES' or json.loads(fixture.read_text())['status']!='PASS_STABLE_TARGET_LOSS_NATIVE_FIXTURE':raise AssertionError('CPU references and Native analytic loss fixture required')
    ref_plan=json.loads((reference.parent/'plan.json').read_text())
    if any(file_sha256(n)!=h for n,h in ref_plan['source_sha256'].items()):raise AssertionError('CPU stable-reference source changed')
    fixture_plan=json.loads((fixture.parent/'plan.json').read_text())
    if file_sha256('compute_trainium/stable_target_loss.py')!=fixture_plan['source_sha256']['compute_trainium/stable_target_loss.py']:raise AssertionError('Native loss fixture source changed')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete snapshot inode allowance')
    for p in [root,Path('/')]:
        if shutil.disk_usage(p).free<37*1024**3:raise RuntimeError('Complete small output and protected disk allowance')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<(512+192)*1024**3:raise RuntimeError('Full192GiB worker plus512GiB reserve')
    old=json.loads((root/'workspace-native-full-update-v1/plan.json').read_text())
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for n in ['scripts/run_native_isolated.sh','tests/test_stable_target_loss.py','tests/test_workspace_native_objective.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);start=time.time();phase=IDENTITY+'-native'
    cases=dict(c0=dict(inputs_relative='workspace-identity-gradient-v1/private-objective-inputs.pt',gradient_relative=ref['c0_gradient_relative']),
        mature=dict(inputs_relative='runs/workspace-native-online-throughput-v1-online/private-mature-inputs.pt',gradient_relative=ref['mature_gradient_relative']))
    for case,value in cases.items():
        value.update(inputs_sha256=file_sha256(root/value['inputs_relative']),gradient_sha256=file_sha256(root/value['gradient_relative']))
    mature_sha=[file_sha256(root/'runs/workspace-native-online-throughput-v1-online'/f'rank{k}-u102.zip') for k in range(64)]
    if mature_sha!=json.loads((root/'workspace-online-mature-cpu-audit-v1/plan.json').read_text())['shard_sha256']:raise AssertionError('Unchanged failed-endpoint model required')
    plan=dict(identity=IDENTITY,purpose='workspace_stable_model_fixed_points',world=64,worker_memory_gib=192,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,registered_unix=start,
        seconds_limit=1800,absolute_deadline_unix=start+1800,phase_seconds_limit=1600,cpu_phase_seconds_limit=60,
        clock_admission_waiver=True,inode_projection=projection,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='fixed')},
        checkpoint_relative=old['checkpoint_relative'],checkpoint_sha256=old['checkpoint_sha256'],model_config=old['model_config'],
        opening_ids=old['opening_ids'],closing_ids=old['closing_ids'],mature_shards_relative='runs/workspace-native-online-throughput-v1-online',
        cases=cases,mature_shard_sha256=mature_sha,reference_report_sha256=file_sha256(reference),fixture_report_sha256=file_sha256(fixture),
        optimizer_updates=0,policy_fit_admitted=False,scope='Requalify unchanged initial/saturated actual1.485B weights with stable FP32 target loss against independent HF-body/FP64-loss full gradients.Frozen reward labels isolate the derivative change.No historical trajectory is repaired.No scientific fit.')
    out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_STABLE_MODEL_FIXED_POINTS',policy_fit_admitted=False)
    def stop(n,f):raise KeyboardInterrupt('Owned fixed-point qualification interrupted')
    old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        phase=IDENTITY+'-native';result['numerical']=execution.native(phase,'scripts.qualify_workspace_stable_model',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--comm-port',str(execution.plan['root_comm_ports'][phase])],
            'PASS_WORKSPACE_STABLE_FIXED_MODEL_POINTS')
        result['status']='PASS_WORKSPACE_STABLE_FIXED_MODEL_POINTS'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(out/'execution-costs.json');result['plan_sha256']=execution.sha
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in old.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
