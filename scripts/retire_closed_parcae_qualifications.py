"""Retire 15 explicitly named successful disposable qualification arrays."""
import argparse,json,shutil,subprocess,time
from pathlib import Path
from scripts.retire_closed_probability_qualifications import digest


def inactive():
    if subprocess.check_output(['docker','ps','-q','--filter','label=ac-trainium-managed=1']).strip():
        raise RuntimeError('All accelerator owners must be closed')


def main(root):
    started=time.time();inactive();out=root/'closed-parcae-disposable-retirement-v1'
    if out.exists():raise FileExistsError('Immutable one-time maintenance')
    gate=Path('results/parcae_low_depth_qualification_20261024/attempt4/report.json')
    evidence=json.loads(gate.read_text())
    if evidence['status']!='PASS_COMPLETE_PARCAE_TRAINING_QUALIFICATION':raise AssertionError('Complete successful qualification required')
    selections=[('fixed1','fp32'),('poisson2','fp16'),('fixed1','fp16')]
    filenames=['after-1-model-optimizer.pt','after-120-full-state.pt','after-121-model-optimizer.pt','initial-global-native.pt','updated-global-native.pt']
    rows=[]
    for arm,precision in selections:
        matches=[r for r in evidence['records'] if r['arm']==arm and r['precision']==precision and r['accumulation']==8]
        if len(matches)!=1:raise AssertionError('Unambiguous successful phase required')
        r=matches[0]
        if not r['passes_recovery_gate'] or r['independent_trained_gradient']['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':raise AssertionError('Qualification failed')
        folder=root/'runs'/f'parcae-low-qual-v4-{precision}-a8-{arm}-warm'
        for filename in filenames:
            p=folder/filename
            if not p.is_file() or p.is_symlink() or p.stat().st_nlink!=1:raise AssertionError('Original singly linked named qualification array required')
            h=digest(p)
            if filename=='after-120-full-state.pt' and h!=r['independent_trained_gradient']['checkpoint_sha256']:raise AssertionError('Audited checkpoint hash differs')
            rows.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h))
    scope='Only15 named successful disposable low-depth qualification arrays are retired. Scientific fits,failed histories,other qualification variants,source data and original audit receipts remain. Local restoration of these removed files is not promised; source/configuration/seed support recomputation without a bit-exact promise.'
    plan=dict(registered_unix=started,seconds_limit=600,files=rows,gate_sha256=digest(gate),source_sha256=digest(Path(__file__)),scope=scope)
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');before=shutil.disk_usage(root).free
    for row in rows:
        inactive();p=root/row['path']
        if time.time()-started>600 or digest(p)!=row['sha256']:raise RuntimeError('Bound or file changed')
        p.with_name(p.name+'.retired.json').write_text(json.dumps(dict(**row,scope=scope,plan=out.name+'/plan.json'),indent=2)+'\n');p.unlink()
    result=dict(status='COMPLETE_CLOSED_PARCAE_QUALIFICATION_RETIREMENT',**plan,freed_bytes=sum(r['bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(root).free,wall_seconds=time.time()-started,scientific_checkpoints_deleted=0,failed_histories_deleted=0,assigned_chips=0)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','freed_bytes','wall_seconds')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
