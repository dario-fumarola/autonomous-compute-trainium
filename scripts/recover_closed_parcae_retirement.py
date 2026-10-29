"""Complete the already registered 15-file retirement after permission failure."""
import argparse,json,shutil,time
from pathlib import Path
from scripts.retire_closed_probability_qualifications import digest
from scripts.retire_closed_parcae_qualifications import inactive


def main(root):
    start=time.time();inactive();base=root/'closed-parcae-disposable-retirement-v1';p=json.loads((base/'plan.json').read_text())
    names=['after-1-model-optimizer.pt','after-120-full-state.pt','after-121-model-optimizer.pt','initial-global-native.pt','updated-global-native.pt']
    expected={f'runs/parcae-low-qual-v4-{precision}-a8-{arm}-warm/{n}' for arm,precision in [('fixed1','fp32'),('poisson2','fp16'),('fixed1','fp16')] for n in names}
    if {r['path'] for r in p['files']}!=expected or len(p['files'])!=15:raise AssertionError('Original exact scope required')
    for row in p['files']:
        f=root/row['path']
        if f.is_symlink() or f.stat().st_nlink!=1 or digest(f)!=row['sha256']:raise AssertionError('Original undeleted bytes required')
    out=base/'recovery-v1';out.mkdir(exist_ok=False)
    registration=dict(registered_unix=start,seconds_limit=600,original_plan_sha256=digest(base/'plan.json'),source_sha256=digest(Path(__file__)),scope=p['scope'],reason='Original attempt failed on first sidecar PermissionError before any deletion; this recovery has scoped filesystem privileges.')
    (out/'plan.json').write_text(json.dumps(registration,indent=2)+'\n');before=shutil.disk_usage(root).free
    for row in p['files']:
        inactive();f=root/row['path']
        if time.time()-start>600 or digest(f)!=row['sha256']:raise RuntimeError('File changed or maintenance cap')
        f.with_name(f.name+'.retired.json').write_text(json.dumps(dict(**row,scope=p['scope'],plan=base.name+'/plan.json'),indent=2)+'\n');f.unlink()
    report=dict(status='COMPLETE_CLOSED_PARCAE_QUALIFICATION_RETIREMENT',**registration,files=p['files'],freed_bytes=sum(r['bytes'] for r in p['files']),free_before=before,free_after=shutil.disk_usage(root).free,wall_seconds=time.time()-start,assigned_chips=0,scientific_checkpoints_deleted=0,failed_histories_deleted=0)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','freed_bytes','wall_seconds')}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root)
