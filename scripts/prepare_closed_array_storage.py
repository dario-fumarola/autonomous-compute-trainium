"""Bounded read-only inventory of byte-identical completed hidden-state arrays."""
import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import resource
import shutil
import time

from compute_trainium.closed_array_storage import array_record, duplicate_actions
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY = 'closed-hidden-array-storage-v1'
PUBLIC = Path('results/closed_hidden_array_storage_20261026')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a=p.parse_args()
    started=time.time(); out=a.root/IDENTITY
    if out.exists() or PUBLIC.exists(): raise FileExistsError('Preserve each storage inventory')
    if managed_running(): raise RuntimeError('No maintenance while a native worker is live')
    out.mkdir(); PUBLIC.mkdir(parents=True)
    files=['scripts/prepare_closed_array_storage.py','compute_trainium/closed_array_storage.py', 'tests/test_closed_array_storage.py']
    registration=dict(identity=IDENTITY,registered_unix=started,source_sha256={f:file_sha256(f) for f in files},
        seconds_limit=600,max_cpu_hash_workers=4,scope='Read-only SHA256 inventory of completed-run h0*.npy arrays. No array contents,models,corpora,source snapshots or logs are removed. A later byte-preserving hard-link operation needs its own explicit bounded receipt and unchanged-source/path/hash checks.',mutations=0,assigned_chips=0)
    write_state(out/'registration.json',registration);write_state(PUBLIC/'registration.json',registration)
    def record(path):
        if time.time()-started>600 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Bounded storage inventory reserve/cap')
        return array_record(a.root,path)
    paths=sorted(p for p in (a.root/'runs').glob('*/h0*.npy') if (p.parent/'report.json').is_file())
    with futures.ThreadPoolExecutor(4) as pool: records=list(pool.map(record,paths))
    actions=duplicate_actions(records)
    # Count physical inodes rather than names; a replaced inode may have an
    # additional pre-existing alias,which prevents immediate space recovery.
    reclaimable={row['duplicate']['inode']:row['duplicate']['size'] for row in actions if row['duplicate']['links']==1}
    plan=dict(identity=IDENTITY,records=records,actions=actions,source_sha256=registration['source_sha256'],
        canonical_paths_must_remain_immutable=True,original_per_path_metadata_preserved=True)
    write_state(out/'plan.json',plan)
    report=dict(status='COMPLETE_READ_ONLY_CLOSED_HIDDEN_ARRAY_INVENTORY',
        plan_sha256=file_sha256(out/'plan.json'),array_paths=len(records),duplicate_paths=len(actions),
        estimated_immediately_reclaimable_bytes=sum(reclaimable.values()),estimated_reclaimable_inodes=len(reclaimable),
        wall_seconds=time.time()-started,cpu_seconds=time.process_time(),peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        mutations=0,model_updates=0,assigned_chips=0,scope=registration['scope'])
    write_state(out/'report.json',report);write_state(PUBLIC/'report.json',report);print(json.dumps(report))


if __name__=='__main__': main()
