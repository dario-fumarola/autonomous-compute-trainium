"""Retire only named disposable software arrays; retain canonical qualifications."""
import argparse,json,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256


def main(root):
    if managed_running():raise RuntimeError('No maintenance over active accelerator owner')
    gate=Path('results/probability_workspace_20261028/batch4-recovery-v1/report.json')
    if json.loads(gate.read_text())['status']!='PASS_PROBABILITY_BATCH4_RECOVERY_AND_MATURE_AUDIT':raise AssertionError('Keep batch1 until batch4 fully qualified')
    out=root/'probability-obsolete-fixture-retirement-v1';out.mkdir();start=time.time();files=[]
    warm=root/'runs/probability-workspace-warm-v1-warm'
    files += [warm/n for n in ('PRIVATE-u103.pt','PRIVATE-u104.pt','PRIVATE-mature-gradient.pt')]
    replay=root/'runs/probability-workspace-warm-v1-replay/PRIVATE-mature-gradient.pt'
    if replay.exists():files.append(replay)
    names=('parcae-native-opt-precise-v1-recurrent-primary','parcae-native-opt-precise-v1-ordinary-primary')
    for name in names:
        folder=root/'runs'/name
        for canonical in ('after-two-full-state.pt.exact.gz','native-step-3.pt.exact.gz'):
            if not (folder/canonical).is_file():raise AssertionError('Canonical optimizer fixtures must remain')
        for n in ('native-step-1.pt','native-step-2.pt'):
            blob=folder/(n+'.exact.gz');recipe=folder/(n+'.exact.json')
            row=json.loads(recipe.read_text())
            # Original archive receipt retained; a sibling retirement receipt
            # explicitly supersedes its restoration availability.
            if not blob.is_file():raise AssertionError('Selected obsolete archive missing')
            files.append(blob)
    records=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=file_sha256(f)) for f in files]
    plan=dict(registered_unix=start,files=records,scope='Superseded batch1 disposable warm arrays and obsolete optimizer step1/2 software fixtures only. Canonical batch4 replay states and optimizer after-two/step3 retained. No scientific model,failed history,source corpus or original receipt deleted.',
        irreversible_effect='These exact intermediate fixture arrays are no longer byte-restorable locally; source/seed/configuration retained for recomputation. This supersedes the earlier exact-archive availability for the four retired blobs.',
        qualifier_sha256=file_sha256(gate))
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');before=shutil.disk_usage(root).free
    for f,row in zip(files,records):
        if managed_running() or time.time()-start>600:raise RuntimeError('Bounded closed-fixture maintenance')
        if file_sha256(f)!=row['sha256']:raise AssertionError('Fixture changed after inventory')
        (f.parent/(f.name+'.retired.json')).write_text(json.dumps(dict(**row,receipt=str(out.name)+'/plan.json',reason=plan['scope'],exact_local_restoration_available=False),indent=2)+'\n')
        f.unlink()
    result=dict(status='COMPLETE_OBSOLETE_SOFTWARE_FIXTURE_RETIREMENT',files=records,scientific_checkpoints_deleted=0,failed_histories_deleted=0,
        freed_bytes=sum(r['bytes'] for r in records),free_before=before,free_after=shutil.disk_usage(root).free,wall_seconds=time.time()-start,assigned_chips=0,
        exact_local_restoration_available_for_retired_intermediates=False,canonical_qualification_states_retained=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
