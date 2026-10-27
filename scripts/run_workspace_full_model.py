"""One bounded all-chip zero-update actual-model gradient comparison."""
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

IDENTITY='workspace-native-full-model-v1'
PUBLIC=Path('results/workspace_native_full_model_20261027')


def prepare(root):
    from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig
    from transformers import AutoConfig
    from dataclasses import asdict
    if managed_running():raise RuntimeError('Exclusive actual-model qualification')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every qualification')
    cpu=root/'workspace-identity-gradient-v1/report.json';proof=json.loads(cpu.read_text())
    tiny=root/'workspace-native-sharded-fixture-v1/report.json'
    if proof['status']!='PASS_WORKSPACE_IDENTITY_FULL_MODEL_GRADIENT' or json.loads(tiny.read_text())['status']!='PASS_WORKSPACE_NATIVE_SHARDED_SOFTWARE_FIXTURE':raise AssertionError('CPU actual-model and Native sharded numerical closure required')
    original=json.loads((root/'workspace-identity-gradient-v1/plan.json').read_text())
    if any(file_sha256(n)!=h for n,h in original['source_sha256'].items()):raise AssertionError('Actual-model CPU reference source changed')
    for path in [root,Path('/')]:
        if shutil.disk_usage(path).free<37*1024**3:raise RuntimeError('Complete2GiB small artifacts plus35GiB reserve')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<(512+128)*1024**3:raise RuntimeError('Complete128GiB worker plus protected host reserve')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete snapshot inode reserve')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for p in ['scripts/run_native_isolated.sh','tests/test_workspace_joint_terms.py']:sources[p]=file_sha256(p)
    worker,host=split_snapshot_sources(sources);start=time.time();phase=IDENTITY+'-gradient'
    reader=json.loads((root/'workspace-reader-qualification-v1/plan.json').read_text())
    gradient='workspace-identity-gradient-v1/gradients.pt';inputs='workspace-identity-gradient-v1/private-objective-inputs.pt';checkpoint='workspace-reader-bootstrap-v1/checkpoint.pt'
    if file_sha256(root/gradient)!=proof['full_gradient_sha256'] or file_sha256(root/inputs)!=proof['objective_inputs_sha256'] or file_sha256(root/checkpoint)!=original['calibrated_checkpoint_sha256']:raise AssertionError('Exact qualified states required')
    config=asdict(OlmoWorkspaceConfig.from_hf(AutoConfig.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True)))
    plan=dict(identity=IDENTITY,purpose='workspace_actual_model_zero_update_gradient',world=64,worker_memory_gib=128,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        registered_unix=start,absolute_deadline_unix=start+1800,seconds_limit=1800,phase_seconds_limit=1500,cpu_phase_seconds_limit=60,
        clock_admission_waiver=True,inode_projection=projection,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='gradient')},
        cpu_report_sha256=file_sha256(cpu),tiny_native_report_sha256=file_sha256(tiny),
        checkpoint_relative=checkpoint,checkpoint_sha256=original['calibrated_checkpoint_sha256'],gradient_relative=gradient,
        gradient_sha256=proof['full_gradient_sha256'],inputs_relative=inputs,inputs_sha256=proof['objective_inputs_sha256'],
        model_config=config,padded_positions=192,opening_ids=reader['opening_ids'],closing_ids=reader['closing_ids'],cost_per_token=.001,
        optimizer_updates=0,policy_fit_admitted=False,scope='Actual1.485B fully sharded FP32 model plus independent frozen scorer,activation checkpointing,selective vocabulary head,sequential exact joint-objective terms.No optimizer update. Full179 named gradients compared with independent HF CPU reference at2% limit. Runtime memory and failures measured. No scientific campaign admitted.')
    out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_WORKSPACE_ACTUAL_MODEL_GRADIENT',optimizer_updates=0,policy_fit_admitted=False)
    def stop(n,f):raise KeyboardInterrupt('Owned actual-model qualification stop')
    previous={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        phase=IDENTITY+'-gradient'
        result['numerical']=execution.native(phase,'scripts.qualify_workspace_full_model',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,
             '--comm-port',str(execution.plan['root_comm_ports'][phase])],'PASS_WORKSPACE_ACTUAL_MODEL_GRADIENT')
        result['status']='PASS_WORKSPACE_ACTUAL_MODEL_GRADIENT'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(out/'execution-costs.json');result['plan_sha256']=execution.sha
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in previous.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
