"""Retire eight named successful disposable warm arrays after the tied gate."""
import json,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    if managed_running():raise RuntimeError('All Native owners must be closed')
    gate=Path('results/interleaved_workspace_20261029/tied-comparison-v1/report.json')
    if not json.loads(gate.read_text())['useful_equal_data_screen']:raise AssertionError('No unnecessary follow-up maintenance before the tied gate')
    root=Path('/run/user/1000/autonomous-compute/instance-nvme');profiles=[
        ('interleaved-latent-warm-v1','results/interleaved_workspace_20261029/latent-warm-v1/report.json','PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY'),
        ('probability-budget-cue-warm-v1','results/probability_workspace_20261029/budget-cue-warm-v1/report.json','PASS_BUDGET_CUE_ACTUAL_SHAPE')]
    rows=[];gates={str(gate):file_sha256(gate)}
    for name,path,status in profiles:
        p=Path(path);q=json.loads(p.read_text())
        if q['status']!=status or q['records']['warm']['retained_for_science'] or q['records']['warm']['scientific_optimizer_updates']!=0:
            raise AssertionError('Completed disposable qualification required')
        gates[path]=file_sha256(p)
        for phase,names in [('warm',('PRIVATE-u103.pt','PRIVATE-u104.pt','PRIVATE-mature-gradient.pt')),('replay',('PRIVATE-mature-gradient.pt',))]:
            for n in names:
                f=root/'runs'/(name+'-'+phase)/n
                if not f.is_file() or f.is_symlink() or f.stat().st_nlink!=1:raise AssertionError('Named single-link regular file required')
                rows.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=file_sha256(f)))
    out=root/'closed-superseded-warm-retirement-v1';out.mkdir(mode=0o700);start=time.time()
    r=dict(status='REGISTERED_CLOSED_SUPERSEDED_WARM_RETIREMENT',files=rows,gate_sha256=gates,source_sha256=file_sha256(__file__),freed_bytes=0,
        scientific_states_deleted=0,failed_histories_deleted=0,assigned_chips=0,exact_local_restoration_after_retirement=False,
        scope='Only eight successful disposable warm/replay arrays for the superseded chronological interleaved graph and closed budget-cue recipe. All scientific states, failed histories, original receipts, configurations, sources and canonical small numerical fixtures remain. Current grouped/mixed/tied qualification arrays remain. Exact removed bytes are no longer locally restorable; recomputation is not a bit-exact promise.')
    write_state(out/'report.json',r)
    try:
        for row in rows:
            if managed_running() or time.time()-start>600:raise RuntimeError('Closed-owner/600second bound')
            f=root/row['path']
            if file_sha256(f)!=row['sha256']:raise AssertionError('Inventory bytes changed')
            write_state(f.with_name(f.name+'.retired.json'),dict(**row,exact_local_restoration=False,scope=r['scope']))
            f.unlink();r['freed_bytes']+=row['bytes'];write_state(out/'report.json',r)
        r['status']='COMPLETE_CLOSED_SUPERSEDED_WARM_RETIREMENT'
    except BaseException as e:
        r.update(status='STOPPED_CLOSED_SUPERSEDED_WARM_RETIREMENT',error_type=type(e).__name__,error=str(e));raise
    finally:r['wall_seconds']=time.time()-start;write_state(out/'report.json',r)


if __name__=='__main__':main()
