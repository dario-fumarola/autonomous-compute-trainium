"""Prospective full-state allowance on existing private RAM/NVMe only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.training import file_sha256
    from scripts.run_scale_sharding_fixture import SCALE_END
    if managed_running():raise RuntimeError('Storage admission follows Native closure')
    tune=json.loads((root/'scale-medium-tune-v1/report.json').read_text())
    if tune['status']!='COMPLETE_INDEPENDENT_SCALE_RATE_SELECTION':raise ValueError('Complete TUNE required before main storage admission')
    rec=root.parent/'recurrence';out=root/'scale-main-state-allowance-v1'
    if out.exists():raise FileExistsError('Preserve every resource admission')
    current=sum(f.stat().st_size for d in ['scale-tune-original-initializers-v1','scale-medium-tune-retained-v1'] for f in (rec/d).glob('*.pt'))
    maximum=26*1024**3;remaining=maximum-current
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if remaining<0 or available-remaining-192*1024**3<512*1024**3:
        raise RuntimeError('Full remaining26GiB task evidence plus192GiB Native and512GiB reserve must fit')
    if time.time()>SCALE_END:raise TimeoutError('Original scale cap')
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(rec)],text=True))['filesystems'][0]
    if mount['fstype']!='tmpfs' or mount['target']!=str(rec):raise ValueError('Exact existing private tmpfs required')
    def size(native,pattern):return sum(f.stat().st_size for f in (root/native/'artifacts/warm').glob(pattern))
    r='scale-376m-fixed-two-numerical-v1';o='scale-375m-ordinary-numerical-v1'
    rs,os_=size(r,'rank*-u100.pt'),size(o,'rank*-u100.pt');rg=size(r,'gradient-u100.pt');og=size(o,'gradient-u100.pt');noise=size(r,'noise-u100-rank*.pt')
    root_allowance=3*rs+rg+noise+4*os_+og+4*1024**3
    equal_allowance=2*os_+og+64*1024**2
    initial=root/'scale-main-original-initializers-v1';initial_report=json.loads((initial/'report.json').read_text())
    if initial_report['phase']!='main' or initial_report['seed']!=49177 or initial_report['optimizer_updates']!=0:raise ValueError('Original unupdated main C0 required')
    entry=next(r for r in initial_report['records'] if r['role']=='fixed_two');source=initial/entry['file']
    if source.is_symlink() or file_sha256(source)!=entry['sha256']:raise AssertionError('Original R C0 changed')
    if current+entry['bytes']+equal_allowance>maximum:raise RuntimeError('Complete Rec data/state inventory exceeds26GiB new-task allowance')
    if shutil.disk_usage(root).free+entry['bytes']<35*1024**3+root_allowance:
        raise RuntimeError('All unique main checkpoints/probes/replays plus4GiB margin must fit NVMe prospectively')
    out.mkdir();plan=dict(started_unix=time.time(),purpose='main_full_state_storage_only',
        preceding16gib_allowance_sha256=file_sha256(root/'scale-evidence-memory-allowance-v1/plan.json'),
        maximum_new_task_ram_evidence_bytes=maximum,existing_new_task_bytes=current,remaining_new_task_allowance=remaining,
        host_available_before=available,projected_full_native_and_evidence_minimum_available=available-remaining-192*1024**3,
        native_worker_allowance_bytes=192*1024**3,protected_host_bytes=512*1024**3,
        root_complete_additional_bytes=root_allowance,rec_equal_data_additional_bytes=equal_allowance,
        old_original_tmpfs_gib=160,new_tmpfs_gib=186,original_scale_deadline_unix=SCALE_END,
        scientific_main_fit_admitted=False,raw_sources_and_existing_states_unchanged=True,
        policy='Main preserves all raw states including replay, even if trajectories differ. Equal-data ordinary endpoint uses existing RAM; primary paths use existing NVMe. All source/compute/scientific clocks unchanged.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run(['sudo','-n','mount','-o','remount,size=186G',str(rec)],check=True)
    destination=rec/'scale-main-original-initializers-v1';destination.mkdir(exist_ok=False)
    copy=destination/entry['file'];shutil.copyfile(source,copy)
    if copy.stat().st_size!=entry['bytes'] or file_sha256(copy)!=entry['sha256']:raise AssertionError('Preserved original main C0 copy differs')
    relative=os.path.relpath(copy,source.parent);source.unlink();source.symlink_to(relative)
    if file_sha256(source)!=entry['sha256']:raise AssertionError('Original logical C0 path differs')
    (rec/'scale-main-equal-data-v1').mkdir(exist_ok=False)
    result=dict(status='ADMITTED_COMPLETE_SCALE_MAIN_STATE_STORAGE',**plan,
        fixed_two_original_logical_path_and_sha256_preserved=True,relative_link_valid_under_existing_container_mounts=True,
        main_initializer_sha256=entry['sha256'],root_free_after=shutil.disk_usage(root).free,rec_free_after=shutil.disk_usage(rec).free,
        finished_unix=time.time())
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.root)),flush=True)
