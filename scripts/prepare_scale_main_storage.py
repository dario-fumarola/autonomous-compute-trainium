"""Prospective complete main-state allowance on existing NVMe, no remounts."""
import fnmatch
import json
from pathlib import Path
import shutil
import time


def original_capture_sizes(folder):
    """Exact original lengths, whether still raw or losslessly retained."""
    sizes={f.name:f.stat().st_size for f in folder.glob('*.pt') if not f.is_symlink()}
    for file in folder.glob('*.pt.exact.json'):
        record=json.loads(file.read_text());name=record['original_file']
        if name in sizes and sizes[name]!=record['original_bytes']:raise AssertionError('Conflicting preserved size inventory')
        sizes[name]=record['original_bytes']
    return sizes


def main_state_allowance(root):
    from compute_trainium.training import file_sha256
    sizes={};proofs={}
    for role,label in [('fixed_two','fixed-two'),('ordinary8','ordinary')]:
        identity='scale-65536-'+label+'-numerical-v1';proof=root/identity/'completion-verification.json'
        if json.loads(proof.read_text())['status']!='PASS_SCALE_65536_COMPLETE_NUMERICAL_QUALIFICATION':raise ValueError('Both complete corrected numerical proofs required')
        proofs[role]=file_sha256(proof)
        rows=original_capture_sizes(root.parent/'recurrence'/identity/'warm')
        def group(pattern,count):
            selected=[v for k,v in rows.items() if fnmatch.fnmatchcase(k,pattern)]
            if len(selected)!=count:raise ValueError('Complete exact original capture-size inventory required: '+pattern)
            return sum(selected)
        sizes[role]=dict(state=group('rank*-u100.pt',64),gradient=group('gradient-u100.pt',1),
            noise=group('noise-u100-rank*.pt',64) if role=='fixed_two' else 0)
    r,o=sizes['fixed_two'],sizes['ordinary8']
    # R: two final prefixes plus actual replay; at most three periodic copies.
    # O: retained trunk,two equal-data states,two work states,actual replay.
    # Three-copy rotating-write peaks fit this same six-copy aggregate bound.
    raw=3*r['state']+r['gradient']+r['noise']+6*o['state']+2*o['gradient']
    return dict(complete_additional_state_bytes=raw,margin_bytes=4*1024**3,
        complete_additional_bytes=raw+4*1024**3,original_sizes=sizes,numerical_proofs=proofs)


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.training import file_sha256
    from scripts.run_scale_sharding_fixture import SCALE_END
    if managed_running():raise RuntimeError('Storage admission follows Native closure')
    tune_file=root/'scale-medium-tune-65536-v1/report.json';tune=json.loads(tune_file.read_text())
    if tune['status']!='COMPLETE_INDEPENDENT_SCALE_RATE_SELECTION':raise ValueError('Complete corrected TUNE required before main admission')
    out=root/'scale-main-state-allowance-v2'
    if out.exists():raise FileExistsError('Preserve every resource admission')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<(512+192)*1024**3:raise RuntimeError('Complete192GiB worker plus protected512GiB required')
    if time.time()+14*3600>SCALE_END:raise TimeoutError('Complete14-hour main allowance must fit original cap')
    allowance=main_state_allowance(root)
    if shutil.disk_usage(root).free<35*1024**3+allowance['complete_additional_bytes']:
        raise RuntimeError('All main state/probe/replay/rotating peaks plus4GiB margin and35GiB reserve must fit')
    if min(shutil.disk_usage('/').free,shutil.disk_usage(root.parent).free)<36*1024**3:raise RuntimeError('Protected auxiliary output reserves')
    initial=root/'scale-main-original-initializers-v1';original=json.loads((initial/'report.json').read_text())
    if original['phase']!='main' or original['seed']!=49177 or original['optimizer_updates']!=0 or original['model_forwards']!=0:
        raise ValueError('Fresh original main C0 required')
    for record in original['records']:
        if file_sha256(initial/record['file'])!=record['sha256']:raise AssertionError('Original main C0 changed')
    out.mkdir();result=dict(status='ADMITTED_COMPLETE_SCALE_MAIN_STATE_STORAGE',started_unix=time.time(),
        purpose='main_complete_existing_nvme_state_allowance',**allowance,corrected_tune_report_sha256=file_sha256(tune_file),
        host_available_before=available,native_worker_allowance_bytes=192*1024**3,protected_host_bytes=512*1024**3,
        root_free_before=shutil.disk_usage(root).free,protected_filesystem_bytes=35*1024**3,
        original_scale_deadline_unix=SCALE_END,new_ram_allowance_bytes=0,remount_performed=False,
        source_or_initializer_migration_performed=False,scientific_main_fit_admitted=False,
        policy='All main endpoints,trunk,distinct replay states and probes retained on existing NVMe. Only prospectively rotating periodic backups can retire after a complete verified successor; their manifests and exact inventories remain. No failure or source release.')
    (out/'plan.json').write_text(json.dumps(result,indent=2)+'\n');(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.root)),flush=True)
