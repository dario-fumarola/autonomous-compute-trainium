"""Register execution recovery and independently verify every adopted state."""
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch

from compute_trainium.low_depth_recovery import rebind_checkpoint
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    started=time.time();torch.set_num_threads(2)
    experiment=a.root/'parcae-low-depth-pilot-v1';old_file=experiment/'main-plan.json'
    old=json.loads(old_file.read_text());old_sha=file_sha256(old_file)
    original_execution=json.loads((experiment/'main-execution/execution-start.json').read_text())
    failure_file=experiment/'main-execution/report.json';failure=json.loads(failure_file.read_text())
    if (failure['status']!='STOPPED_FROZEN_LOW_DEPTH_MAIN_PILOT'
            or failure['error']!='Explicit safe phase identity and half required'
            or original_execution['plan_sha256']!=old_sha):raise ValueError('Exact preserved failed execution required')
    if time.time()>original_execution['deadline_unix']:raise TimeoutError('Original main window expired')
    if any(file_sha256(f)!=sha for f,sha in old['source_sha256'].items()):raise AssertionError('Original frozen science changed')
    plan=copy.deepcopy(old);plan['identity']='parcae-low-depth-main-recovery-v2'
    files=['scripts/prepare_low_depth_main_recovery.py','scripts/run_low_depth_main_recovery.py',
           'compute_trainium/low_depth_recovery.py','tests/test_low_depth_recovery.py']
    plan['source_sha256'].update({f:file_sha256(f) for f in files})
    plan['worker_source_sha256']={f:sha for f,sha in plan['source_sha256'].items() if f.startswith(('scripts/','compute_trainium/','configs/'))}
    identities=[f'low-depth-main-r2-{arm}-{phase}' for arm in ['ordinary','fixed1','poisson2'] for phase in ['first','finish','equal']]
    plan['root_comm_ports']=reserve_ports(a.root,identities)
    plan['recovery']={'registered_unix':started,'original_plan_sha256':old_sha,
        'original_failure_sha256':file_sha256(failure_file),'original_execution_deadline_unix':original_execution['deadline_unix'],
        'changes':'Lowercase gradient execution identities,new owned launch identities/ports,metadata-only checkpoint-plan signature rebinding. No scientific configuration changes.',
        'ordinary_restart':'Failed70-update prefix has no saved checkpoint;preserve it and its costs,restart only ordinary from its identical registered fresh initialization.',
        'predecessor_gradient_gates':'low-depth-recovery-v2-gradient-gates'}
    plan_file=experiment/'main-plan-recovery-v2.json';public=Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2')
    if plan_file.exists():raise FileExistsError('No silent recovery registration replacement')
    public.mkdir(parents=True,exist_ok=False);write_state(plan_file,plan);write_state(public/'plan.json',plan)
    new_sha=file_sha256(plan_file);adoptions=[]
    for arm in ['fixed1','poisson2']:
        origin=a.root/'runs'/f'low-depth-main-v1-{arm}-first'
        report_file=origin/'report.json';report=json.loads(report_file.read_text())
        if report['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or report['update']!=1920 or report['plan_sha256']!=old_sha:
            raise ValueError('Completed exact1B predecessor required')
        target=a.root/'runs'/f'low-depth-main-r2-{arm}-first';target.mkdir(exist_ok=False)
        adopted=copy.deepcopy(report);adopted['plan_sha256']=new_sha
        adopted['adopted_from']={'identity':origin.name,'report_sha256':file_sha256(report_file),'checkpoints':{}}
        adopted['new_optimizer_calls_this_adoption']=0
        adopted['execution_mode']='adopted_verified_complete_predecessor_segment'
        for row in adopted['checkpoint_records']:
            old_checkpoint=origin/f'checkpoint-u{row["update"]}.pt';original_sha=row['checkpoint_sha256']
            if file_sha256(old_checkpoint)!=original_sha or not row['all_complete_replica_states_equal']:
                raise AssertionError('Original checkpoint/all-replica proof changed')
            saved=torch.load(old_checkpoint,map_location='cpu',mmap=True,weights_only=False)
            before=digest({k:v for k,v in saved.items() if k!='signature'})
            rebound=rebind_checkpoint(saved,old_sha,old,new_sha,plan,report['case_id'])
            file=target/old_checkpoint.name;atomic_checkpoint(file,rebound)
            readback=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
            if digest({k:v for k,v in readback.items() if k!='signature'})!=before:
                raise AssertionError('Adoption changed model,optimizer,RNG,schedule or source/work state')
            if digest({'model':dict(sorted(readback['model'].items())),'optimizer':readback['optimizer']})!=row['full_state_digest']:
                raise AssertionError('Adopted complete model/optimizer digest differs')
            row['checkpoint_sha256']=file_sha256(file)
            adopted['adopted_from']['checkpoints'][str(row['update'])]={'original_sha256':original_sha,
                'adopted_sha256':row['checkpoint_sha256'],'non_signature_payload_digest':before,'all_non_signature_state_exact':True}
            del saved,rebound,readback
        write_state(target/'report.json',adopted)
        adoptions.append({'arm':arm,'source_report_sha256':file_sha256(report_file),
                          'checkpoints':adopted['adopted_from']['checkpoints']})
    out=experiment/'main-execution-recovery-v2';out.mkdir(exist_ok=False)
    write_state(out/'execution-start.json',{**original_execution,'plan_sha256':new_sha,
        'recovery_registered_unix':started,'original_started_unix':original_execution['started_unix']})
    receipt={'status':'PASS_METADATA_ONLY_MAIN_CHECKPOINT_ADOPTION','records':adoptions,
        'original_deadline_preserved':True,'all_scientific_state_exact':True,'assigned_chips':0,
        'optimizer_updates':0,'wall_seconds':time.time()-started,'test_accessed':False,
        'scope':'All tensors,optimizer moments,RNG,source cursor,LR schedule and work counters are bit-exact. Only plan signature changes for new owned identities/ports. Original failed artifacts and costs remain. No new scientific intervention or reset deadline.'}
    write_state(out/'adoption.json',receipt);write_state(public/'adoption.json',receipt)
    print(json.dumps({'status':receipt['status'],'seconds':receipt['wall_seconds'],'original_deadline_unix':original_execution['deadline_unix']}))


if __name__=='__main__':main()
