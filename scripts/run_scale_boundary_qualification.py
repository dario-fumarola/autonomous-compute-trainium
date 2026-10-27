"""Fresh boundary-precision qualification; no scientific fit or clock renewal."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from scripts import run_scale_model_warm as engine

IDENTITY='scale-boundary-fixed-two-numerical-v1'
PUBLIC=Path('results/scale_boundary_numerical_20261027')


def prepare(root):
    if managed_running():raise RuntimeError('Prior Native owner must be closed')
    diagnosis=root/'scale-reproduced-boundary-followup-v1/report.json'
    observation=json.loads(diagnosis.read_text())
    if (observation['status']!='COMPLETE_REPRODUCED_BOUNDARY_DIAGNOSIS'
            or not observation['comparison']['passes_original_thresholds']):
        raise ValueError('Passing unchanged-point boundary observation required')
    parent_file=root/'scale-65536-fixed-two-numerical-v1/plan.json';parent=json.loads(parent_file.read_text())
    tune=json.loads((root/'scale-medium-tune-65536-v1/plan.json').read_text())
    start=time.time()
    if start+2700>tune['absolute_deadline_unix']:raise TimeoutError('Complete45-minute allowance must fit original TUNE cap')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every registration and failure')
    allowance=parent['warm_artifact_allowance_bytes']+parent['replay_fallback_allowance_bytes']+2*1024**3
    if shutil.disk_usage(root).free<35*1024**3+allowance:raise RuntimeError('Complete warm/replay/unshared fallback allowance plus35GiB required')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<704*1024**3:raise RuntimeError('192GiB worker plus512GiB host reserve required')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for f in ['scripts/run_native_isolated.sh','tests/test_scale_precision_probe.py','docs/SCALE_BOUNDARY_QUALIFICATION_20261027.md']:sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-'+m for m in ['warm','replay']]
    artifact=folder/'artifacts';container='/work/instance-nvme/'+IDENTITY+'/artifacts'
    plan={**parent,'identity':IDENTITY,'registered_unix':start,'source_sha256':sources,'worker_source_sha256':worker,
        'host_only_source_sha256':host,'precision_policy':'fp32_recurrent_boundaries','manual_loss_scale':65536,
        'host_artifact_root':str(artifact),'container_artifact_root':container,
        'host_artifact_roots':dict(warm=str(artifact),replay=str(artifact)),
        'container_artifact_roots':dict(warm=container,replay=container),
        'phases':{n:dict(mode=m) for n,m in zip(names,['warm','replay'])},'root_comm_ports':reserve_ports(root,names),
        'seconds_limit':2700,'phase_seconds_limit':600,'cpu_phase_seconds_limit':1200,
        'absolute_deadline_unix':start+2700,'expected_artifact_peak_bytes':allowance,
        'boundary_diagnosis_sha256':file_sha256(diagnosis),'parent_qualification_plan_sha256':file_sha256(parent_file),
        'scope':Path('docs/SCALE_BOUNDARY_QUALIFICATION_20261027.md').read_text()}
    folder.mkdir();artifact.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    folder=root/IDENTITY;plan=json.loads((folder/'plan.json').read_text());child=None
    result=dict(status='STOPPED_BOUNDARY_NUMERICAL_QUALIFICATION',scientific_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('Bounded numerical qualification stopped')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.run(root)
        capture=json.loads((folder/'report.json').read_text())
        if capture['status']!='CAPTURED_SCALE_ACTUAL_SHAPE_CPU_AUDITS_PENDING':raise RuntimeError('Incomplete warm/replay capture')
        cpu=folder/'independent-cpu';cpu.mkdir();registration=cpu/'plan.json'
        write_state(registration,dict(identity=IDENTITY+'-independent-cpu',native_identity=IDENTITY,
            purpose='independent_scale_numerical_cpu_only',registered_unix=time.time(),
            absolute_deadline_unix=min(time.time()+1200,plan['absolute_deadline_unix']),source_sha256=plan['source_sha256'],
            native_plan_sha256=file_sha256(folder/'plan.json'),native_capture_report_sha256=file_sha256(folder/'report.json'),
            maximum_cpu_optimizer_steps=1,maximum_global_objective_windows=512,assigned_chips=0,scientific_fit_admitted=False))
        env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'24','MKL_NUM_THREADS':'24','PYTHONPATH':str(Path.cwd())}
        with (cpu/'controller.log').open('x') as log:
            child=subprocess.Popen(['/home/ubuntu/ema-trainium/.venv/bin/python','-B','-m','scripts.audit_scale_model',
                '--root',str(root),'--output',str(cpu/'audit'),'--registration',str(registration)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child.wait(timeout=min(1200,plan['absolute_deadline_unix']-time.time()))
        if child.returncode:raise RuntimeError('Independent CPU audit failed; preserve complete evidence')
        audit=json.loads((cpu/'audit/report.json').read_text())
        if (audit['status']!='PASS_SCALE_ACTUAL_SHAPE_INDEPENDENT_CPU_AUDITS'
                or not audit['first_update']['passes'] or not audit['full512']['gradient']['passes_two_percent_gate']
                or abs(audit['full512']['native_minus_cpu_nll'])>.003 or not audit['replay']['passes']):
            raise AssertionError('Every original independent gate must pass')
        result.update(status='PASS_BOUNDARY_COMPLETE_NUMERICAL_QUALIFICATION',manual_loss_scale=65536,
            precision_policy=plan['precision_policy'],actual_qualification_updates=102,
            first_update_moments_pass=True,full512=audit['full512'],replay=audit['replay'],
            independent_cpu_report_sha256=file_sha256(cpu/'audit/report.json'),warm=capture['records']['warm'])
        write_state(PUBLIC/'independent-audit.json',audit)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result.update(plan_sha256=file_sha256(folder/'plan.json'),finished_unix=time.time())
        write_state(folder/'completion-verification.json',result);write_state(PUBLIC/'completion-verification.json',result)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
