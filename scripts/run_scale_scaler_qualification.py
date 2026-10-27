"""Actual-shape qualification of65536 scaling; original attempts untouched."""
import argparse
import json
from pathlib import Path
import shutil
import time

from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import SCALE_END
from scripts import run_scale_model_warm as engine


def prepare(root,role,identity,public):
    if managed_running():raise RuntimeError('Prior Native owners must be terminal')
    probe=root/'scale-trained-precision-probe-v1/report.json';diagnosis=json.loads(probe.read_text())
    if (diagnosis['status']!='COMPLETE_UNCHANGED_SCALE_PRECISION_DIAGNOSIS'
            or not diagnosis['records']['scaled16']['passes_unchanged_point_gate']):raise ValueError('Passing unchanged-point scaler diagnosis required')
    if role=='ordinary8':
        proof=json.loads((root/'scale-65536-fixed-two-numerical-v1/completion-verification.json').read_text())
        if proof['status']!='PASS_SCALE_65536_COMPLETE_NUMERICAL_QUALIFICATION':raise ValueError('Complete higher-scale recurrent gate required first')
    parent=root/('scale-376m-fixed-two-numerical-v1' if role=='fixed_two' else 'scale-375m-ordinary-numerical-v1')
    original=json.loads((parent/'plan.json').read_text());folder=root/identity;rec=root.parent/'recurrence'/identity
    if folder.exists() or rec.exists() or public.exists():raise FileExistsError('Preserve all numerical attempts')
    old=parent/'artifacts/warm'
    state_bytes=sum(f.stat().st_size for f in old.glob('rank*-u100.pt'))
    gradient_bytes=(old/'gradient-u100.pt').stat().st_size
    noise_bytes=sum(f.stat().st_size for f in old.glob('noise-u100-rank*.pt'))
    warm_allowance=3*state_bytes+2*gradient_bytes+noise_bytes+512*1024**2
    if shutil.disk_usage(rec.parent).free<35*1024**3+warm_allowance:raise RuntimeError('Complete actual warm-state/gradient/noise allowance plus reserve required')
    if shutil.disk_usage(root).free<35*1024**3+state_bytes+512*1024**2:
        raise RuntimeError('Complete unshared replay fallback must fit, even if exact sharing fails')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for f in ['scripts/run_native_isolated.sh','tests/test_scale_qualification_sharing.py','tests/test_scale_precision_probe.py']:
        sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);names=[identity+'-'+m for m in ['warm','replay']];start=time.time()
    if start+3600+7200>SCALE_END:raise TimeoutError('Complete numerical and CPU allowances do not fit original scale cap')
    folder.mkdir();rec.mkdir();(folder/'artifacts').mkdir();public.mkdir(parents=True)
    plan={**original,'identity':identity,'registered_unix':start,'source_sha256':sources,'worker_source_sha256':worker,
        'host_only_source_sha256':host,'manual_loss_scale':65536,'capture_initial_noise':False,
        'replay_capture_gradients':False,'share_exact_replay':True,
        'host_artifact_root':str(rec),'container_artifact_root':'/work/recurrence/'+identity,
        'host_artifact_roots':dict(warm=str(rec),replay=str(folder/'artifacts')),
        'container_artifact_roots':dict(warm='/work/recurrence/'+identity,replay='/work/instance-nvme/'+identity+'/artifacts'),
        'phases':{n:dict(mode=m) for n,m in zip(names,['warm','replay'])},'root_comm_ports':reserve_ports(root,names),
        'seconds_limit':3600,'phase_seconds_limit':1800,'absolute_deadline_unix':min(start+3600,SCALE_END),
        'expected_artifact_peak_bytes':warm_allowance+state_bytes,'warm_artifact_allowance_bytes':warm_allowance,
        'replay_fallback_allowance_bytes':state_bytes+512*1024**2,
        'preceding_failed_science_preserved':True,'precision_diagnosis_sha256':file_sha256(probe),
        'parent_actual_shape_plan_sha256':file_sha256(parent/'plan.json'),
        'scope':'Qualification only from original unupdated random seed1009 C0, same accepted6.5B prefix/context1024/global512/all64 cores. Only scaler65536 replaces8192; model,loss,precision operators and optimizer math unchanged.101 initial updates include100 warm intervals; one fresh-process nonzero-rate replay. Full independent first-update/moments/full512 gradient/NLL/replica/RNG/replay gates required. No initial noise copy is needed for the observed-gradient first-update isolation; trained100 noise is retained. Exact live replays may share immutable warm bytes only after all64 complete-state digests match; any difference retains every raw rank. Original failed157M history is ineligible, never repaired retroactively. All clocks stay inside the original24-hour scale cap. No quality/science fit admitted.'}
    write_state(folder/'plan.json',plan);write_state(public/'plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['fixed_two','ordinary8'],required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    identity='scale-65536-'+('fixed-two' if a.role=='fixed_two' else 'ordinary')+'-numerical-v1'
    public=Path('results/scale_65536_numerical_20261027')/a.role
    engine.IDENTITY=identity;engine.PUBLIC=public
    engine.run(a.root) if a.run else prepare(a.root,a.role,identity,public)
