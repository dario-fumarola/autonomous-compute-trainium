"""Fresh complete storage admission for the frozen boundary development point."""
import json,shutil,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.execution_isolation import managed_running
from scripts.prepare_scale_main_storage import main_state_allowance
from scripts.run_scale_sharding_fixture import SCALE_END


def prepare(root):
    if managed_running():raise RuntimeError('Storage admission follows Native closure')
    from scripts.run_scale_boundary_lower_rates import selection_source
    tune_plan_path,tune_path=selection_source(root);tune=json.loads(tune_path.read_text())
    if tune['status'] not in ['COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION','COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION']:raise ValueError('Complete boundary rate selection required')
    scope_path=root/'scale-boundary-scope-freeze-v1/scope.json';scope=json.loads(scope_path.read_text())
    tune_plan=json.loads(tune_plan_path.read_text())
    if tune_plan['scope_freeze_sha256']!=file_sha256(scope_path):raise AssertionError('Frozen scope changed')
    amendment=root/'research-clock-accounting-amendment-20261027-v1/plan.json'
    clock=json.loads(amendment.read_text())
    if not clock['overall_clock_admission_waived'] or not clock['scientific_endpoints_unchanged']:
        raise ValueError('Explicit owner clock amendment required')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<704*1024**3:raise RuntimeError('Full192GiB worker and512GiB host reserve required')
    proof=root/'scale-boundary-fixed-two-numerical-v1/completion-verification.json'
    if json.loads(proof.read_text())['status']!='PASS_BOUNDARY_COMPLETE_NUMERICAL_QUALIFICATION':raise ValueError('New recurrent boundary proof required')
    # Same tensor inventories; complete measured raw sizes plus4GiB cover
    # lineage/precision metadata as well as every unshared replay fallback.
    allowance=main_state_allowance(root)
    if shutil.disk_usage(root).free<35*1024**3+allowance['complete_additional_bytes']:raise RuntimeError('Full raw state/probe/replay/periodic peaks,margin and reserve must fit')
    if min(shutil.disk_usage('/').free,shutil.disk_usage(root.parent).free)<36*1024**3:raise RuntimeError('Protected auxiliary outputs plus1GiB cache margin')
    initial=root/'scale-main-original-initializers-v1';c0=json.loads((initial/'report.json').read_text())
    if c0['phase']!='main' or c0['seed']!=49177 or c0['optimizer_updates']!=0 or c0['model_forwards']!=0:raise ValueError('Original untouched main C0 required')
    for row in c0['records']:
        if file_sha256(initial/row['file'])!=row['sha256']:raise AssertionError('Main C0 bytes changed')
    out=root/'scale-boundary-main-state-allowance-v1'
    if out.exists():raise FileExistsError('Preserve every resource admission')
    out.mkdir();value=dict(status='ADMITTED_COMPLETE_SCALE_MAIN_STATE_STORAGE',registered_unix=time.time(),**allowance,
        boundary_numerical_proof_sha256=file_sha256(proof),tuning_report_sha256=file_sha256(tune_path),scope_freeze_sha256=file_sha256(scope_path),
        actual_root_free_before=shutil.disk_usage(root).free,host_available_before=available,native_worker_allowance_bytes=192*1024**3,
        protected_host_bytes=512*1024**3,protected_filesystem_bytes=35*1024**3,new_ram_allowance_bytes=0,
        main_required_seconds=scope['main_required_seconds'],original_scale_deadline_unix=SCALE_END,scientific_main_fit_admitted=False,
        user_clock_amendment_sha256=file_sha256(amendment),overall_clock_admission_waived=True,
        scope='Complete existing-NVMe raw-state allowance,including all distinct replay fallback and periodic-write peaks. No source/state deletion or remount. Main preparation must recheck current full free space. Original qualifying sizes are exact-byte retained;new boundary model has identical parameter/state inventories,with4GiB metadata/size margin.')
    (out/'plan.json').write_text(json.dumps(value,indent=2)+'\n');(out/'report.json').write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.root)),flush=True)
