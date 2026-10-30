"""Consolidate six byte-identical old recovery pairs, preserving both paths."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='closed-recovery-copy-consolidation-v1'
PUBLIC=Path('results/closed_recovery_copy_consolidation_20261030')


def main(root):
    start=time.time(); recovery=Path('/home/ubuntu/autonomous-compute-recovery')
    receipt=recovery/'mechanism-copy-receipt.json'
    accepted=json.loads(receipt.read_text())
    if accepted['status']!='PASS_SIX_NEW_FULL_STATE_EBS_RECOVERY_COPIES' or len(accepted['copies'])!=6:
        raise AssertionError('Six previously verified complete recovery copies required')
    if managed_running(): raise RuntimeError('No Native owner may be active')
    out=root/ID
    if out.exists() or PUBLIC.exists(): raise FileExistsError('Preserve maintenance attempts')
    out.mkdir(mode=0o700); PUBLIC.mkdir(parents=True)
    pairs=[]
    for row in accepted['copies']:
        run=row['run']; disk=root/'runs'/run/'checkpoint.pt'; backup=recovery/(run+'.pt')
        closure=root/'snapshots'/run/'closure.json'
        if json.loads(closure.read_text()).get('exit_code')!=0:
            raise AssertionError('Only successful terminal old runs')
        remove,retain=(disk,backup) if run.startswith('untied-depth-v1-') else (backup,disk)
        for path in (remove,retain):
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink!=1 or path.stat().st_size!=row['bytes']:
                raise AssertionError('Two independent regular exact-size copies required')
        pairs.append(dict(remove=str(remove),retain=str(retain),bytes=row['bytes'],sha256=row['sha256'],
                          closure=str(closure),closure_sha256=file_sha256(closure)))
    scope=('Consolidate six successful October23 full-state duplicate pairs after verifying BOTH complete files '
           'against their original recovery hashes. Keep one unchanged physical copy and atomically replace the other '
           'path by a symbolic link. Four root-filesystem duplicates and two NVMe duplicates are released. '
           'Both original paths remain byte-readable; six pairs no longer provide two-filesystem redundancy. '
           'No unique model, optimizer state, RNG state, failed attempt, source, audit or dataset is deleted. '
           'No off-instance redundancy is claimed.')
    plan=dict(identity=ID,registered_unix=start,seconds_limit=600,source_sha256=file_sha256(__file__),
              original_recovery_receipt_sha256=file_sha256(receipt),pairs=pairs,scope=scope)
    write_state(out/'PRIVATE-plan.json',plan)
    # No machine-specific physical paths in the public aggregate.
    write_state(PUBLIC/'plan.json',{k:v for k,v in plan.items() if k!='pairs'})
    before=dict(root=shutil.disk_usage('/').free,nvme=shutil.disk_usage(root).free)
    done=[]; report=dict(status='STOPPED_CLOSED_RECOVERY_COPY_CONSOLIDATION')

    def guard():
        if managed_running() or time.time()-start>600:
            raise RuntimeError('Closed-owner / duration bound')

    try:
        # Verify all pairs before mutating any path.
        for row in pairs:
            guard()
            if any(file_sha256(row[k])!=row['sha256'] for k in ('remove','retain')):
                raise AssertionError('Original full recovery bytes must agree')
        for row in pairs:
            guard(); remove,retain=Path(row['remove']),Path(row['retain'])
            if file_sha256(row['closure'])!=row['closure_sha256']:
                raise AssertionError('Terminal receipt changed')
            old_remove,old_retain=remove.stat(),retain.stat()
            # Single-owner closed maintenance; recheck contents immediately
            # before replacing the duplicate directory entry.
            if file_sha256(remove)!=row['sha256'] or file_sha256(retain)!=row['sha256']:
                raise AssertionError('Copy changed before consolidation')
            temporary=remove.with_name(remove.name+'.consolidation-link')
            if temporary.exists() or temporary.is_symlink(): raise FileExistsError('No overwritten temporary link')
            temporary.symlink_to(retain)
            os.replace(temporary,remove)
            if (not remove.is_symlink() or remove.resolve()!=retain.resolve()
                    or retain.stat().st_ino!=old_retain.st_ino or file_sha256(remove)!=row['sha256']):
                raise AssertionError('Original path must read unchanged retained bytes')
            record=dict(bytes=row['bytes'],sha256=row['sha256'],released_filesystem='nvme' if old_remove.st_dev==root.stat().st_dev else 'root')
            done.append(record)
            write_state(out/'progress.json',dict(completed=done))
        report['status']='COMPLETE_CLOSED_RECOVERY_COPY_CONSOLIDATION'
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error)); raise
    finally:
        report.update(files=len(done),freed_bytes=sum(r['bytes'] for r in done),records=done,
                      free_before=before,free_after=dict(root=shutil.disk_usage('/').free,nvme=shutil.disk_usage(root).free),
                      wall_seconds=time.time()-start,assigned_chips=0,unique_scientific_states_deleted=0,
                      original_paths_byte_readable=True,dual_filesystem_redundancy_reduced=True,
                      plan_sha256=file_sha256(out/'PRIVATE-plan.json'),scope=scope)
        write_state(out/'report.json',report); write_state(PUBLIC/'report.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
