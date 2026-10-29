"""Recover space while preserving exact bytes and both old qualification paths."""
import json,os,stat,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    if managed_running():raise RuntimeError('Closed Native owner boundary required')
    root=Path('/run/user/1000/autonomous-compute/instance-nvme')
    receipt=Path('results/owner_full_training_screen_20261025/report.json')
    old=json.loads(receipt.read_text())
    if old['status']!='COMPLETE_OWNER_FULL_TRAINING_SCREEN' or not old['qualification_only'] or old['research_optimizer_updates']!=0 or old['scientific_adoption']:
        raise AssertionError('Only closed, non-adopted qualification paths')
    start=time.time();rows=[]
    for name,key in [('after-first-update.pt','first_update_sha256'),('after121.pt','final_sha256')]:
        paths={k:root/'runs'/('owner-full-training-screen-v1-one-ntp-'+k)/name for k in ('owner','replicated')}
        hashes={k:old['records'][k][key] for k in paths}
        if len(set(hashes.values()))!=1:raise AssertionError('Original receipts must record identical bytes')
        infos={k:p.lstat() for k,p in paths.items()}
        for k,p in paths.items():
            if not stat.S_ISREG(infos[k].st_mode) or infos[k].st_nlink!=1 or file_sha256(p)!=hashes[k]:
                raise AssertionError('Exact named single-link regular snapshot required')
        if infos['owner'].st_dev!=infos['replicated'].st_dev:raise AssertionError('Same filesystem required')
        if any(getattr(infos['owner'],a)!=getattr(infos['replicated'],a) for a in ('st_uid','st_gid','st_mode','st_size')):
            raise AssertionError('Preserve owner, permission and size semantics')
        rows.append(dict(paths={k:str(p.relative_to(root)) for k,p in paths.items()},sha256=hashes['owner'],
                         bytes=infos['owner'].st_size,duplicate_allocated_bytes=infos['replicated'].st_blocks*512))
    out=root/'closed-owner-snapshot-deduplication-v1';out.mkdir(mode=0o700)
    result=dict(status='REGISTERED_EXACT_CLOSED_SNAPSHOT_DEDUPLICATION',files=rows,
        gate_sha256=file_sha256(receipt),source_sha256=file_sha256(__file__),assigned_chips=0,
        scientific_states_deleted=0,failed_histories_deleted=0,freed_allocated_bytes=0,
        scope='Two byte-identical pairs from the closed October25 owner/replicated throughput screen. Preserve all paths, exact serialized bytes, receipts and the unfavorable result. Atomic hard links share storage; treat these closed historical snapshots as immutable and copy before any future write. Differing full-state and gradient files remain unchanged.')
    write_state(out/'report.json',result)
    try:
        for row in rows:
            if managed_running() or time.time()-start>600:raise RuntimeError('Closed owner and600second bound')
            keep=root/row['paths']['replicated'];replace=root/row['paths']['owner']
            if any(file_sha256(p)!=row['sha256'] for p in (keep,replace)):raise AssertionError('Snapshot changed')
            temporary=replace.with_name(replace.name+'.dedup-link-pending')
            if temporary.exists():raise AssertionError('No unexpected pending link')
            os.link(keep,temporary)
            try:os.replace(temporary,replace)
            finally:
                if temporary.exists():temporary.unlink()
            if not os.path.samefile(keep,replace) or file_sha256(replace)!=row['sha256']:raise AssertionError('Exact post-link verification')
            result['freed_allocated_bytes']+=row['duplicate_allocated_bytes'];write_state(out/'report.json',result)
        result['status']='COMPLETE_EXACT_CLOSED_SNAPSHOT_DEDUPLICATION'
    except BaseException as error:
        result.update(status='STOPPED_EXACT_CLOSED_SNAPSHOT_DEDUPLICATION',error_type=type(error).__name__,error=str(error));raise
    finally:
        result['wall_seconds']=time.time()-start;write_state(out/'report.json',result)


if __name__=='__main__':main()
