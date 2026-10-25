"""Register and execute a finite original-conclusion driver gate."""
import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import signal
from pathlib import Path
import time

from compute_trainium.future_span_execution import Execution
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-chain-driver-qualification-v1'
PUBLIC=Path('results/natural_chain_driver_qualification_20261025')


def register(root):
    import torch
    from compute_trainium.natural_chain_objective import bridge_visible
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.parcae_training_candidate import initial_state
    from compute_trainium.training import atomic_checkpoint
    from scripts.qualify_native import digest
    torch.set_num_threads(8)
    if managed_running():raise RuntimeError('Idle native ownership required')
    source=root/'natural-chain-views-v1';previous=json.loads((source/'report.json').read_text())
    if previous['status']!='COMPLETE_TRAIN_ONLY_ORIGINAL_CONCLUSION_VIEWS' or file_sha256(source/'training-views.json')!=previous['views_sha256']:raise ValueError('Immutable original conclusion views required')
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'future-span-main-v2/plan.json').read_text());config=ParcaeCandidateConfig(**manifest['model'])
    parent=root/'runs/future-span-main-v2-one-ntp-s5231-lr32000-finish';report=json.loads((parent/'report.json').read_text())
    checkpoint=parent/'checkpoint-u1920.pt';expected=report['checkpoint_records'][-1]['checkpoint_sha256']
    if file_sha256(checkpoint)!=expected:raise ValueError('Completed common NTP checkpoint changed')
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    state={k:v for k,v in saved['model'].items() if not k.startswith('future_lm_head.') and k!='future_positions'}
    model=ParcaePrimitiveSDPACandidate(config,recurrent=True,initialize=False,fp16_body=True);model.load_state_dict(state,strict=True)
    model_digest=digest(dict(sorted(model.state_dict().items())))
    atomic_checkpoint(out/'initial-body.pt',dict(model=model.state_dict(),model_digest=model_digest,parent_sha256=expected,pretrained_targets=1004887590,optimizer_reset=True,qualification_only=True))
    del saved,state,model
    views=json.loads((source/'training-views.json').read_text());grouped={}
    for row in views:grouped.setdefault(row['document_id'],{})[row['visible_steps']]=row
    chosen=[];groups=set()
    for key in sorted(grouped,key=lambda x:hashlib.sha256(('chain-driver-v1:'+x).encode()).hexdigest()):
        if grouped[key][0]['group'] in groups:continue
        chosen.append(key);groups.add(grouped[key][0]['group'])
        if len(chosen)==32:break
    if len(chosen)!=32:raise ValueError('32 distinct source groups required')
    tasks={k:[] for k in ['inputs','targets','answer_mask','termination_mask','metadata']}
    for update in [0,40,80]:
        rows=[grouped[key][bridge_visible(max(grouped[key]),update)] for key in chosen]
        for k in ['inputs','targets','answer_mask','termination_mask']:tasks[k].append([r[k] for r in rows])
        tasks['metadata'].append([{k:r[k] for k in ['document_id','group','prefix_ids','answer_ids','visible_steps']} for r in rows])
    for k in ['inputs','targets','answer_mask','termination_mask']:tasks[k]=torch.tensor(tasks[k],dtype=torch.bool if k.endswith('mask') else torch.long)
    tasks['h0']=torch.cat([initial_state(config,i,16033)[:,:128] for i in range(32)])
    atomic_checkpoint(out/'tasks.pt',tasks)
    old=json.loads((root/'local-batch-qualification-v2/plan.json').read_text())
    additions=['scripts/run_natural_chain_qualification.py','scripts/qualify_natural_chain_native.py','scripts/audit_natural_chain_cpu.py',
        'compute_trainium/natural_chain_views.py','compute_trainium/natural_chain_objective.py','tests/test_natural_chain_views.py',
        'tests/test_natural_chain_objective.py','docs/NATURAL_CHAIN_DRIVER_QUALIFICATION_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in additions}}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise AssertionError('Qualified predecessor or task source changed')
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-'+k+'-'+m for k in ['one','recurrent'] for m in ['warm','replay']]
    plan=dict(identity=IDENTITY,registered_unix=time.time(),purpose='qualification_original_conclusion_driver',qualification_only=True,
        model=manifest['model'],parent_checkpoint_sha256=expected,parent_targets=1004887590,optimizer_reset=True,
        initial_sha256=file_sha256(out/'initial-body.pt'),model_digest=model_digest,tasks_sha256=file_sha256(out/'tasks.pt'),views_sha256=previous['views_sha256'],
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        context=128,examples=32,source_groups=32,seconds_limit=1800,phase_seconds_limit=600,cpu_phase_seconds_limit=600,
        root_comm_ports=reserve_ports(root,names),scope=Path('docs/NATURAL_CHAIN_DRIVER_QUALIFICATION_20261025.md').read_text(),
        test_accessed=False,terminal_alpha_spent=0,scientific_training_admitted=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);return plan


def run(root,plan):
    out=root/IDENTITY;file=out/'plan.json';execution=Execution(root,file,out/'execution');result=dict(status='STOPPED_NATURAL_CHAIN_DRIVER_QUALIFICATION');records=[];cpu=cf.ThreadPoolExecutor(4)
    def interrupted(signum,frame):
        execution.failed.set();raise KeyboardInterrupt('Finite task qualification interrupted: '+str(signum))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def native(arm,mode,parent=None):
        name=IDENTITY+'-'+arm+'-'+mode
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,
            '--case',arm,'--mode',mode,'--comm-port',str(plan['root_comm_ports'][name])]
        if parent:opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/full-state-u120.pt','--resume-sha',parent[1]['full_checkpoint_sha256']]
        d=execution.native(name,'scripts.qualify_natural_chain_native',opts,'CAPTURED_NATURAL_CHAIN_DRIVER_QUALIFICATION')
        if not d['all_complete_replica_states_equal'] or d['signature']['plan_sha256']!=execution.sha:raise AssertionError('Task identity/replicas differ')
        if parent and (not d['full_state_on_load_exact'] or not d['rng_on_load_exact'] or d['recovered_state_digest']!=parent[1]['full_checkpoint_state_digest']):raise AssertionError('Task full-state replay load failed')
        return name,d
    def audit(arm,mode,update,capture):
        target=out/'cpu'/arm/(mode+'-u'+str(update));opts=['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,'--capture',str(root/'runs'/capture[0]),'--update',str(update),'--output',str(target)]
        return execution.cpu(arm+'-'+mode+'-u'+str(update),'scripts.audit_natural_chain_cpu',opts,target/'report.json','PASS_INDEPENDENT_NATURAL_CHAIN_AUDIT')
    def case(arm):
        warm=native(arm,'warm');first=cpu.submit(audit,arm,'warm',0,warm);trained=cpu.submit(audit,arm,'warm',120,warm)
        replay=native(arm,'replay',warm);again=cpu.submit(audit,arm,'replay',120,replay)
        return arm,warm,replay,[first,trained,again]
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_TRAIN_ONLY_TASK_DRIVER_QUALIFICATION',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as workers:
            for future in cf.as_completed([workers.submit(case,arm) for arm in ['one','recurrent']]):
                arm,warm,replay,checks=future.result()
                records.append(dict(case=arm,native=dict(warm=warm[1],replay=replay[1]),independent_audits=[f.result() for f in checks],
                    exact_future_trajectory_equal=warm[1]['final_model_optimizer_digest']==replay[1]['final_model_optimizer_digest']))
                write_state(PUBLIC/'progress.json',dict(records=records))
        execution.check();result=dict(status='PASS_NATURAL_CHAIN_DRIVER_QUALIFICATION',records=records,actual_qualification_updates=244,scientific_training_admitted=False)
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for sig,handler in handlers.items():signal.signal(sig,handler)
        result.update(plan_sha256=execution.sha,qualification_only=True,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    plan=register(a.root)
    if not a.prepare_only:run(a.root,plan)


if __name__=='__main__':main()
