"""One finite engineering qualification after the scientific owner closes."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.bucketed_benchmark import ORDER,summarize
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='bucketed-native-qualification-v1'
PUBLIC=Path('results/gradient_bucketing_20261030/native-v1')


def prepare(root):
    if managed_running():raise RuntimeError('Active scientific owner must close first')
    current=Path('results/latent_exposure_20261030/work-recovery-v1/report.json')
    if not current.exists() or json.loads(current.read_text())['status'].startswith('RUNNING'):
        raise RuntimeError('Complete current scientific controller before registration')
    gate=Path('results/gradient_bucketing_20261030/mature-cpu-v2/report.json')
    if json.loads(gate.read_text())['status']!='PASS_BUCKETED_MATURE_CPU':raise AssertionError('Mature CPU parity required')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<39*1024**3:
        raise RuntimeError('Protected reserve plus complete4GiB artifact allowance')
    parent=root/'interleaved-tied-ordinary-qualification-v1/plan.json'
    p=copy.deepcopy(json.loads(parent.read_text()))
    data=root/p['data_relative'];report=json.loads((data/'report.json').read_text())
    if file_sha256(data/'report.json')!=p['data_report_sha256']:
        raise AssertionError('Accepted source report changed')
    for name in ('train-tokens.npy','train-domains.npy'):
        if file_sha256(data/name)!=report['files'][name]:raise AssertionError('Accepted TRAIN bytes changed')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for name in ('scripts/run_native_isolated.sh','docs/BUCKETED_NATIVE_QUALIFICATION_20261030.md',
                 'tests/test_bucketed_gradient_reduction.py','tests/test_bucketed_benchmark.py'):
        files[name]=file_sha256(name)
    worker,host=split_snapshot_sources(files);names=[ID+'-'+v for v in ('warm','replay')]
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete inode projection required')
    now=time.time();p.update(identity=ID,purpose='bucketed_gradient_disposable_qualification',
        registered_unix=now,absolute_deadline_unix=now+5400,seconds_limit=5400,
        phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,worker_memory_gib=192,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        trial_order=ORDER,bucket_bytes=32*1024**2,warm_phase=names[0],early_warm_mean_seconds_limit=1.,
        cpu_gate_sha256=file_sha256(gate),completed_science_report_sha256=file_sha256(current),
        qualification_only=True,scientific_fit_admitted=False,retained_for_science=False,
        scientific_optimizer_updates=0,automatic_promotion=False,
        worker_entry='scripts.qualify_bucketed_native',scope=Path('docs/BUCKETED_NATIVE_QUALIFICATION_20261030.md').read_text())
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve all attempts')
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_NOT_STARTED',scientific_fit_admitted=False))


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution')
    result=dict(status='STOPPED_BUCKETED_NATIVE_QUALIFICATION',records={},scientific_fit_admitted=False)
    try:
        for mode in ('warm','replay'):
            name=ID+'-'+mode;args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                '--plan-sha',e.sha,'--mode',mode,'--comm-port',str(e.plan['root_comm_ports'][name])]
            if mode=='replay':args+=['--parent',f'/work/instance-nvme/runs/{ID}-warm']
            status='COMPLETE_BUCKETED_ABBA_PENDING_AUDIT' if mode=='warm' else 'PASS_BUCKETED_EXACT_REPLAY'
            result['records'][mode]=e.native(name,'scripts.qualify_bucketed_native',args,status)
            write_state(PUBLIC/'progress.json',result)
        dest=out/'cpu-audit'
        result['records']['audit']=e.cpu(ID+'-audit','scripts.audit_bucketed_native',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(dest)],
            dest/'report.json','PASS_BUCKETED_NATIVE_COMPLETE_CPU_AUDIT')
        result['timing']=summarize(result['records']['warm']['trials'])
        result['status']='PASS_BUCKETED_NATIVE_NUMERICAL_AND_SPEED' if result['timing']['speed_gate_passed'] else 'PASS_BUCKETED_NATIVE_NUMERICAL_NO_SPEED_PROMOTION'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result['automatic_promotion']=False;write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
