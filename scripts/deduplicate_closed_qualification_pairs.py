"""Two named byte-identical historical qualification pairs, no history deletion."""
import json,os,stat,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

PAIRS=[(('head-free-fixed2-numerical-v2-recovery-half-a-replay','head-free-fixed2-numerical-v2-recovery-half-b-replay'),
        'after-121-model-optimizer.pt','final_checkpoint_sha256'),
       (('parcae-runtime-scalars-v1-ordinary-warm','parcae-low-qual-v4-fp16-a16-ordinary-warm'),
        'after-1-model-optimizer.pt','first_checkpoint_sha256')]


def main():
    root=Path('/run/user/1000/autonomous-compute/instance-nvme');start=time.time()
    if managed_running():raise RuntimeError('Closed Native owner boundary required')
    rows=[]
    for names,filename,key in PAIRS:
        entries=[]
        for name in names:
            folder=root/'runs'/name;receipt=folder/'report.json';closure=root/'snapshots'/name/'closure.json'
            r=json.loads(receipt.read_text())
            if not r['qualification_only'] or r['research_optimizer_updates']!=0 or json.loads(closure.read_text())['exit_code']!=0:
                raise AssertionError('Closed qualification-only original receipts required')
            path=folder/filename;s=path.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or file_sha256(path)!=r[key]:raise AssertionError('Original exact single-link bytes required')
            entries.append(dict(path=str(path.relative_to(root)),sha256=r[key],bytes=s.st_size,allocated=s.st_blocks*512,
                identity=(s.st_dev,s.st_uid,s.st_gid,s.st_mode,s.st_size),receipt_sha256=file_sha256(receipt),closure_sha256=file_sha256(closure)))
        if entries[0]['sha256']!=entries[1]['sha256'] or entries[0]['identity']!=entries[1]['identity']:
            raise AssertionError('Both paths must retain identical bytes and ownership/permissions')
        entries.sort(key=lambda e:(e['allocated'],e['path']))
        rows.append(dict(keep=entries[0],replace=entries[1]))
    out=root/'closed-qualification-snapshot-deduplication-v2';out.mkdir(mode=0o700)
    result=dict(status='REGISTERED_EXACT_QUALIFICATION_PAIR_DEDUPLICATION',pairs=rows,source_sha256=file_sha256(__file__),
        assigned_chips=0,scientific_states_deleted=0,failed_histories_deleted=0,freed_allocated_bytes=0,
        scope='Two named closed qualification pairs. Exact bytes, both paths, ownership and permissions remain; atomic hard links share immutable historical storage. Copy before any future write. No source, model state, optimizer state, receipt or unfavorable outcome is removed.')
    write_state(out/'report.json',result)
    try:
        for row in rows:
            if managed_running() or time.time()-start>600:raise RuntimeError('Closed owner/600second bound')
            keep=root/row['keep']['path'];replace=root/row['replace']['path'];h=row['keep']['sha256']
            if any(file_sha256(p)!=h for p in (keep,replace)):raise AssertionError('Closed bytes changed')
            tmp=replace.with_name(replace.name+'.dedup-link-pending')
            if tmp.exists():raise AssertionError('Unexpected pending link')
            os.link(keep,tmp)
            try:os.replace(tmp,replace)
            finally:
                if tmp.exists():tmp.unlink()
            if not os.path.samefile(keep,replace) or file_sha256(replace)!=h:raise AssertionError('Exact post-link verification')
            result['freed_allocated_bytes']+=row['replace']['allocated'];write_state(out/'report.json',result)
        result['status']='COMPLETE_EXACT_QUALIFICATION_PAIR_DEDUPLICATION'
    except BaseException as error:
        result.update(status='STOPPED_EXACT_QUALIFICATION_PAIR_DEDUPLICATION',error_type=type(error).__name__,error=str(error));raise
    finally:
        result['wall_seconds']=time.time()-start;write_state(out/'report.json',result)


if __name__=='__main__':main()
