"""Retire ten named successful, disposable warm-qualification arrays only."""
import argparse
import json
import shutil
import time
from pathlib import Path
from scripts.retire_closed_probability_qualifications import digest
from scripts.retire_closed_parcae_qualifications import inactive


def main(root):
    start=time.time();inactive()
    out=root/'closed-chain-warm-qualification-retirement-v1'
    if out.exists():raise FileExistsError('Immutable one-time maintenance')
    gate=Path('results/natural_chain_driver_qualification_20261025/recovery-v3/report.json')
    evidence=json.loads(gate.read_text())
    if (evidence['status']!='PASS_NATURAL_CHAIN_DRIVER_QUALIFICATION' or
        not evidence['qualification_only'] or evidence['scientific_training_admitted'] or evidence['research_optimizer_updates']):
        raise AssertionError('Only successful disposable qualification')
    rows=[]
    for case in ('one','recurrent'):
        records=[r for r in evidence['records'] if r['case']==case]
        if len(records)!=1:raise AssertionError('Unambiguous completed case')
        record=records[0];warm=record['native']['warm'];replay=record['native']['replay']
        if (not warm['all_complete_replica_states_equal'] or not replay['full_state_on_load_exact'] or
            not replay['rng_on_load_exact'] or any(r['status']!='PASS_INDEPENDENT_NATURAL_CHAIN_AUDIT' for r in record['independent_audits'])):
            raise AssertionError('Independent gates and restoration required')
        folder=root/'runs'/f'natural-chain-driver-qualification-v3-{case}-warm'
        closure=root/'snapshots'/folder.name/'closure.json'
        if json.loads(closure.read_text())['exit_code']!=0:raise AssertionError('Completed owned closure')
        expected={'full-state-u120.pt':warm['full_checkpoint_sha256'],
            'after-first-update.pt':warm['first_update_sha256'],'after121.pt':warm['final_sha256']}
        expected.update({r['gradient_file']:r['gradient_sha256'] for r in warm['probes']})
        if set(expected)!={'full-state-u120.pt','after-first-update.pt','after121.pt','gradient-u0.pt','gradient-u120.pt'}:
            raise AssertionError('Exact five-file warm scope')
        for name,h in expected.items():
            p=folder/name
            if not p.is_file() or p.is_symlink() or p.stat().st_nlink!=1 or digest(p)!=h:
                raise AssertionError('Original named audited bytes required')
            rows.append(dict(path=str(p.relative_to(root)),sha256=h,bytes=p.stat().st_size))
    scope=('Only ten named successful disposable natural-chain warm qualification arrays are retired. '
           'Scientific checkpoints, failed attempts, replay qualification arrays, source data and all '
           'original receipts remain. Exact local restoration of the ten removed binary files is '
           'not promised. Recorded source/configuration/seed support recomputation without a bit-exact promise.')
    plan=dict(registered_unix=start,seconds_limit=600,files=rows,gate_sha256=digest(gate),
        source_sha256=digest(Path(__file__)),scope=scope,exact_local_restoration_after_retirement=False)
    out.mkdir(mode=0o700);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    before=shutil.disk_usage(root).free
    for row in rows:
        inactive();p=root/row['path']
        if time.time()-start>600 or digest(p)!=row['sha256']:raise RuntimeError('Maintenance bound or file changed')
        p.with_name(p.name+'.retired.json').write_text(json.dumps(dict(**row,scope=scope,plan=out.name+'/plan.json'),indent=2)+'\n')
        p.unlink()
    result=dict(status='COMPLETE_CLOSED_CHAIN_WARM_QUALIFICATION_RETIREMENT',**plan,
        freed_bytes=sum(r['bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(root).free,
        wall_seconds=time.time()-start,scientific_checkpoints_deleted=0,failed_histories_deleted=0,assigned_chips=0)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','freed_bytes','wall_seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
