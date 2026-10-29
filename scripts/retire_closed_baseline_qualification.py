"""Retire only named successful disposable October26 qualification arrays."""
import json,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

PHASES={'mixing':'baseline-numerical-v1-mixing-warm','one':'baseline-numerical-v1-one-warm',
        'ordinary8':'baseline-numerical-recovery-v1-ordinary8-warm'}
FILES={'after-1-model-optimizer.pt':'first_checkpoint_sha256',
       'after-121-model-optimizer.pt':'final_checkpoint_sha256',
       'after-120-full-state.pt':'checkpoint_sha256',
       'initial-global-native.pt':'initial_global_gradient_sha256',
       'updated-global-native.pt':'updated_global_gradient_sha256'}


def main():
    if managed_running():raise RuntimeError('Close all managed Native owners before maintenance')
    root=Path('/run/user/1000/autonomous-compute/instance-nvme')
    gate=Path('results/baseline_numerical_recovery_20261026/report.json');g=json.loads(gate.read_text())
    if g['status']!='PASS_THREE_BASELINE_NUMERICAL_RECOVERY_PROFILES' or g['research_optimizer_updates']!=0 or g['scientific_fit_admitted']:
        raise AssertionError('Completed disposable qualification only')
    selected=[]
    for label,phase in PHASES.items():
        row=g['records'][label];warm=row['warm'];folder=root/'runs'/phase
        if not warm['qualification_only'] or warm['research_optimizer_updates']!=0:
            raise AssertionError('No retained scientific initialization')
        if row['independent_cpu']['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT' or row['restored']['status']!='PASS_BASELINE_RECOVERY_NAMED_REPLAY_AND_GLOBAL_GRADIENT':
            raise AssertionError('Independent mature and exact replay gates')
        actual=json.loads((folder/'report.json').read_text())
        if actual['checkpoint_sha256']!=warm['checkpoint_sha256']:raise AssertionError('Exact historical phase identity')
        for name,key in FILES.items():
            f=folder/name
            if f.is_symlink() or not f.is_file() or f.stat().st_nlink!=1 or file_sha256(f)!=warm[key]:
                raise AssertionError('Named regular file must match original recorded bytes: '+name)
            selected.append(dict(path=str(f.relative_to(root)),sha256=warm[key],bytes=f.stat().st_size))
    if managed_running():raise RuntimeError('Native owner appeared before retirement')
    out=root/'closed-baseline-qualification-retirement-v1';out.mkdir(mode=0o700);start=time.time()
    result=dict(status='REGISTERED_CLOSED_BASELINE_QUALIFICATION_RETIREMENT',files=selected,gate_sha256=file_sha256(gate),
        source_sha256=file_sha256(__file__),freed_bytes=0,assigned_chips=0,scientific_states_deleted=0,failed_histories_deleted=0,
        exact_local_restoration_after_retirement=False,
        scope='Only15 named successful disposable warm-qualification binary arrays are retired after original hashes,independent mature gradients and exact replay gates. All scientific models/optimizer states,failed histories,source data,replay arrays,source snapshots,configurations,seeds and receipts remain. Exact byte restoration of removed arrays is not promised; qualification can be recomputed without a bit-exact promise.')
    write_state(out/'report.json',result)
    try:
        for row in selected:
            if managed_running() or time.time()-start>600:raise RuntimeError('Closed-owner or600second bound')
            f=root/row['path']
            if file_sha256(f)!=row['sha256']:raise AssertionError('Retirement bytes changed')
            write_state(f.with_name(f.name+'.retired.json'),dict(**row,gate_sha256=result['gate_sha256'],exact_local_restoration=False))
            f.unlink();result['freed_bytes']+=row['bytes'];write_state(out/'report.json',result)
        result['status']='COMPLETE_CLOSED_BASELINE_QUALIFICATION_RETIREMENT'
    except BaseException as e:
        result.update(status='STOPPED_CLOSED_BASELINE_QUALIFICATION_RETIREMENT',error_type=type(e).__name__,error=str(e));raise
    finally:
        result['wall_seconds']=time.time()-start;write_state(out/'report.json',result)


if __name__=='__main__':main()
