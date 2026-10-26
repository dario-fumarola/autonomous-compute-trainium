"""Atomic byte-preserving sharing of the explicitly inventoried closed arrays."""
import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import resource
import signal
import shutil
import time
import traceback

from compute_trainium.closed_array_storage import share_identical_array
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='closed-hidden-array-storage-v1'
PUBLIC=Path('results/closed_hidden_array_storage_20261026')


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.root/IDENTITY;inventory=read(out/'report.json');plan=read(out/'plan.json')
    if inventory['status']!='COMPLETE_READ_ONLY_CLOSED_HIDDEN_ARRAY_INVENTORY' or file_sha256(out/'plan.json')!=inventory['plan_sha256']:
        raise AssertionError('Completed exact read-only inventory required')
    registration=out/'sharing-registration.json'
    if not registration.exists():
        receipt=dict(identity=IDENTITY+'-sharing',registered_unix=time.time(),
            inventory_plan_sha256=inventory['plan_sha256'],inventory_report_sha256=file_sha256(out/'report.json'),
            operator_sha256=file_sha256(__file__),sources=plan['source_sha256'],seconds_limit=900,
            maximum_array_aliases=len(plan['actions']),candidate_bytes=inventory['estimated_immediately_reclaimable_bytes'],
            immutable_completed_run_paths_only=True,original_metadata_recorded_in_private_inventory=True,
            preserved=['every original array byte/path/hash','source/corpus arrays','model/optimizer/checkpoints','receipts/logs','source snapshots'],
            physical_storage_operation='Replace duplicate regular h0 NumPy files atomically with hard links to byte-identical completed arrays of the same device,owner,group and mode;original independent inode/mtime recorded.No content is rewritten.',
            model_updates=0,assigned_chips=0)
        write_state(registration,receipt);write_state(PUBLIC/'sharing-registration.json',receipt)
    receipt=read(registration)
    if a.prepare_only:return
    if (out/'sharing-started.json').exists():raise FileExistsError('Preserve interrupted/completed maintenance;no automatic retry')
    if managed_running():raise RuntimeError('Native owners must be closed before storage maintenance')
    if file_sha256(__file__)!=receipt['operator_sha256'] or any(file_sha256(f)!=h for f,h in receipt['sources'].items()):
        raise AssertionError('Verified storage operator changed')
    if file_sha256(out/'plan.json')!=receipt['inventory_plan_sha256'] or file_sha256(out/'report.json')!=receipt['inventory_report_sha256']:
        raise AssertionError('Original inventory identity changed')
    started=time.time();cpu=time.process_time();before=shutil.disk_usage(a.root).free
    write_state(out/'sharing-started.json',dict(unix=started,registration_sha256=file_sha256(registration)))
    completed=[];result=dict(status='STOPPED_CLOSED_HIDDEN_ARRAY_SHARING')
    def interrupted(number,frame):raise KeyboardInterrupt('Bounded storage operation interrupted: '+str(number))
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    def guard():
        if time.time()-started>900 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Bounded byte-preserving maintenance cap/reserve')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3:raise RuntimeError('Protected memory reserve')
    try:
        # Each bounded batch has independent duplicate paths;canonicals stay
        # immutable and may acquire additional links without changing bytes.
        with futures.ThreadPoolExecutor(4) as pool:
            for left in range(0,len(plan['actions']),32):
                guard()
                if managed_running():raise RuntimeError('Unexpected new native owner;stop maintenance')
                for row in pool.map(lambda action:share_identical_array(a.root,action),plan['actions'][left:left+32]):
                    completed.append(row)
                write_state(out/'sharing-progress.json',dict(completed=completed,wall_seconds=time.time()-started))
        guard()
        def verify(row):
            path=a.root/row['path']
            if path.stat().st_size!=row['size'] or file_sha256(path)!=row['sha256']:
                raise AssertionError('An original array path lost its exact content')
            return True
        with futures.ThreadPoolExecutor(4) as pool:
            verified=sum(pool.map(verify,plan['records']))
        result.update(status='COMPLETE_BYTE_PRESERVING_CLOSED_ARRAY_SHARING',all_original_array_paths_verified=verified,
            all_original_array_hashes_unchanged=verified==len(plan['records']))
    except BaseException as error:
        result.update(error=str(error),error_kind=type(error).__name__,traceback=traceback.format_exc());raise
    finally:
        for number,handler in handlers.items():signal.signal(number,handler)
        write_state(out/'sharing-completed-private.json',dict(completed=completed))
        result.update(completed_alias_replacements=len(completed),wall_seconds=time.time()-started,
            cpu_seconds=time.process_time()-cpu,observed_free_bytes_increase=shutil.disk_usage(a.root).free-before,
            final_free_bytes=shutil.disk_usage(a.root).free,registration_sha256=file_sha256(registration),
            model_updates=0,model_forwards=0,assigned_chips=0,
            scope='Immutable completed numerical h0 arrays only;every path/byte hash preserved,original metadata recorded in inventory. Models,optimizers,source corpora,checkpoints,source snapshots and receipts are not modified. Sharing depends on closed run identities remaining immutable. Filesystem-space change is measured separately from the conservative candidate total.')
        write_state(out/'sharing-report.json',result);write_state(PUBLIC/'sharing-report.json',result)
        print(json.dumps(result))


if __name__=='__main__':main()
