"""Bounded five-endpoint mixed NTP/bridge pilot with first-update integrity gates."""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import random
import signal
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-chain-continuation-pilot-v1'
PUBLIC=Path('results/natural_chain_continuation_pilot_20261025')


def register(root):
    import torch
    from compute_trainium.natural_chain_mixed import update_matrix_work
    from compute_trainium.natural_chain_objective import task_depth
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.parcae_work_accounting import muon_update_matmuls
    from compute_trainium.training import PackedCorpus,atomic_checkpoint
    from scripts.qualify_native import digest
    torch.set_num_threads(8)
    if managed_running():raise RuntimeError('Predecessor native jobs must be terminal')
    ancestor=root/'natural-chain-baseline-readout-v1';old=json.loads((ancestor/'plan.json').read_text());prior=json.loads((ancestor/'report.json').read_text())
    if prior['status']!='COMPLETE_ORIGINAL_CHAIN_BASELINE_READOUT':raise ValueError('Completed fixed before-training baseline required')
    source=root/'natural-chain-views-v1';source_report=json.loads((source/'report.json').read_text())
    if file_sha256(source/'training-views.json')!=source_report['views_sha256']:raise AssertionError('Original source views changed')
    science=json.loads((root/'future-span-main-v2/plan.json').read_text());cfg=ParcaeCandidateConfig(**science['model'])
    checkpoint=root/old['cases']['one-s5231']['checkpoint'];expected=old['cases']['one-s5231']['checkpoint_sha256']
    if file_sha256(checkpoint)!=expected:raise AssertionError('Original common One body changed')
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);body={k:v for k,v in saved['model'].items() if not k.startswith('future_lm_head.') and k!='future_positions'}
    model=ParcaePrimitiveSDPACandidate(cfg,recurrent=True,initialize=False,fp16_body=True);model.load_state_dict(body,strict=True)
    initial_digest=digest(dict(sorted(model.state_dict().items())));optimizer_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    atomic_checkpoint(out/'initial-body.pt',dict(model=model.state_dict(),parent_sha256=expected,purpose='natural_chain_pilot_initializer',qualification_only=False,parent_targets=1004887590,optimizer_reset=True))
    del model,body,saved
    cases={}
    for key,arm,weight in [('one-bridge','one',.3),('recurrent-bridge','recurrent',.3),('one-ntp','one',0.),('recurrent-ntp','recurrent',0.)]:
        work=sum(update_matrix_work(cfg,task_depth(arm,u),task_weight=weight,optimizer_per_replica=optimizer_ops) for u in range(200))
        cases[key]=dict(arm=arm,task_weight=weight,total_updates=200,peak_lr=.001,expected_total_matrix_ops=work)
    budget=cases['recurrent-bridge']['expected_total_matrix_ops'];one_update=update_matrix_work(cfg,1,task_weight=.3,optimizer_per_replica=optimizer_ops);long_updates=budget//one_update;shortfall=(budget-long_updates*one_update)/budget
    if long_updates<=200 or shortfall>=.01:raise ValueError('Eligible stronger One matched-work endpoint unavailable')
    cases['one-bridge-work']=dict(arm='one',task_weight=.3,total_updates=long_updates,peak_lr=.001,parent_case='one-bridge',expected_total_matrix_ops=long_updates*one_update,relative_work_shortfall=shortfall,next_update_would_exceed_budget=(long_updates+1)*one_update>budget)
    views=json.loads((source/'training-views.json').read_text());grouped={}
    for index,row in enumerate(views):grouped.setdefault(row['document_id'],{})[row['visible_steps']]=index
    problems=sorted(grouped);view_indices=[]
    for stage in [0,1,2]:
        view_indices.append([grouped[key][max(grouped[key]) if stage==0 else max(grouped[key])//2 if stage==1 else 0] for key in problems])
    generator=torch.Generator(device='cpu').manual_seed(40009);order=[]
    while len(order)<long_updates*128:order.extend(torch.randperm(len(problems),generator=generator).tolist())
    tasks={k:torch.tensor([row[k] for row in views],dtype=torch.bool if k.endswith('mask') else torch.long) for k in ['inputs','targets','answer_mask','termination_mask']}
    tasks.update(view_indices=torch.tensor(view_indices,dtype=torch.long),order=torch.tensor(order[:long_updates*128],dtype=torch.long).view(long_updates,128),metadata=[{k:r[k] for k in ['document_id','group','prefix_ids','answer_ids','visible_steps']} for r in views]);atomic_checkpoint(out/'tasks.pt',tasks)
    data=science['data'];packed=PackedCorpus.from_mapped(root/data['packed'],1024,23,data['corpus_sha256']);start=1920*512
    if start+long_updates*128>packed.windows or packed.fingerprint!=data['packing_fingerprint']:raise ValueError('Correct append-only original NTP stream required')
    from compute_trainium.natural_chain_pilot_data import answer_count
    for key,case in cases.items():
        case['expected_ntp_targets']=sum(packed.counts(start,case['total_updates']*128));case['expected_answer_targets']=sum(answer_count(tasks,u,case) for u in range(case['total_updates'])) if case['task_weight'] else 0
    del tasks,packed
    extras=['scripts/run_natural_chain_pilot.py','scripts/train_natural_chain_pilot.py','scripts/audit_natural_chain_mixed_cpu.py','scripts/score_natural_chain_retention_cpu.py',
        'compute_trainium/natural_chain_mixed.py','compute_trainium/natural_chain_pilot_data.py','tests/test_natural_chain_mixed.py','tests/test_natural_chain_pilot_data.py','docs/NATURAL_CHAIN_CONTINUATION_PILOT_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in extras}};worker,host=split_snapshot_sources(sources)
    names=[IDENTITY+'-'+key+'-'+mode for key in cases for mode in (['fit'] if key=='one-bridge-work' else ['preflight','fit'])]+[IDENTITY+'-readout-'+key for key in cases]
    plan=dict(identity=IDENTITY,registered_unix=time.time(),purpose='natural_chain_curriculum_continuation_pilot',source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        model=science['model'],cases=cases,initial_sha256=file_sha256(out/'initial-body.pt'),initial_model_digest=initial_digest,parent_checkpoint_sha256=expected,parent_targets=1004887590,parent_nominal_matrix_ops=528283816731279360,
        tasks_sha256=file_sha256(out/'tasks.pt'),source_views_sha256=source_report['views_sha256'],sampling_seed=40009,unique_problems=len(problems),optimizer_matrix_ops_per_replica=optimizer_ops,
        data=data,ntp_start_cursor=start,noise_seed=16033,readout=dict(panel=old['panel'],panel_sha256=old['panel_sha256'],noise=old['noise'],noise_sha256=old['noise_sha256'],primary_recurrent_depth=2,maximum_new_tokens=24),
        retention=science['evaluation'],seconds_limit=7200,phase_seconds_limit=1200,cpu_phase_seconds_limit=1200,
        root_comm_ports=reserve_ports(root,names),budget=dict(maximum_warm_seconds_per_update=2.,scope='Finite200-update screening plus prospective matched-work control. Measured component gates bound feasibility;actual100 mixed warm updates are recorded before any larger follow-up.'),
        scope=Path(extras[-1]).read_text(),test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);return plan


def analyze(root,records):
    by_case={r['case']:r for r in records};all_rows={}
    for key,rec in by_case.items():
        folder=root/'runs'/(IDENTITY+'-readout-'+key);rows=json.loads((folder/'records.json').read_text());all_rows[key]={(r['document_id'],r['depth']):r for r in rows}
    base=all_rows['one-bridge'];ids=sorted(k[0] for k in base);groups={}
    for identifier in ids:groups.setdefault(base[(identifier,1)]['group'],[]).append(identifier)
    def correct(key,depth,identifier):return float(all_rows[key][(identifier,depth)]['correct'])
    contrasts={
        'primary_recurrence_over_same_curriculum':lambda i:correct('recurrent-bridge',2,i)-correct('one-bridge',1,i),
        'factorial_interaction':lambda i:(correct('recurrent-bridge',2,i)-correct('one-bridge',1,i))-(correct('recurrent-ntp',2,i)-correct('one-ntp',1,i)),
        'same_weight_supported_depth_1_to_2':lambda i:correct('recurrent-bridge',2,i)-correct('recurrent-bridge',1,i),
        'recurrence_over_longer_one_work_control':lambda i:correct('recurrent-bridge',2,i)-correct('one-bridge-work',1,i)}
    effects={};keys=sorted(groups)
    for label,fn in contrasts.items():
        values={key:[sum(fn(i) for i in members),len(members)] for key,members in groups.items()};rng=random.Random(1327);draws=[]
        for _ in range(3000):
            chosen=rng.choices(keys,k=len(keys));draws.append(sum(values[g][0] for g in chosen)/sum(values[g][1] for g in chosen))
        draws.sort();effect=sum(fn(i) for i in ids)/len(ids);effects[label]=dict(accuracy_difference=effect,conditional_source_group_95_interval=[draws[74],draws[2924]],observed_at_least_five_percentage_points=effect>=.05)
    return dict(effects=effects,scope='One shared pretrained seed/one paired adaptation stream on consumed discovery DEV. Conditional source-group uncertainty is not training-seed confirmation. No terminal success,best-exit selection or alpha spent.')


def run(root,plan):
    out=root/IDENTITY;plan_file=out/'plan.json';execution=Execution(root,plan_file,out/'execution');result=dict(status='STOPPED_NATURAL_CHAIN_CONTINUATION_PILOT');prefits={};fits={};records=[];cpu=cf.ThreadPoolExecutor(4)
    def interrupted(signum,frame):execution.failed.set();raise KeyboardInterrupt('Mixed pilot interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def native(key,mode,parent=None):
        name=IDENTITY+'-'+key+'-'+mode;opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',key,'--mode',mode,'--comm-port',str(plan['root_comm_ports'][name])]
        if parent:opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/'+parent[1]['checkpoint_file'],'--resume-sha',parent[1]['checkpoint_sha256']]
        target='CAPTURED_MIXED_PILOT_INTEGRATION' if mode=='preflight' else 'COMPLETE_MIXED_CURRICULUM_PILOT_FIT';row=execution.native(name,'scripts.train_natural_chain_pilot',opts,target)
        if row['case']!=key or row['plan_sha256']!=execution.sha or not row['all_replicas_equal']:raise AssertionError('Mixed pilot case/replica identity differs')
        if parent and (not row['full_state_on_load_exact'] or not row['cpu_rng_on_load_exact'] or row['recovered_state_digest']!=parent[1]['full_state_digest']):raise AssertionError('Complete mixed pilot restore failed')
        if mode=='fit':
            case=plan['cases'][key]
            if row['update']!=case['total_updates'] or row['ntp_exposure']!=case['expected_ntp_targets'] or row['answer_token_exposures']!=case['expected_answer_targets'] or row['model_matrix_ops']+row['optimizer_matrix_ops_all_replicas']!=case['expected_total_matrix_ops']:raise AssertionError('Mixed source/task/work endpoint counters differ')
        return name,row
    def integration(key,capture):
        target=out/'cpu'/'integration'/key;opts=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,'--case',key,'--capture',str(root/'runs'/capture[0]),'--output',str(target)]
        return execution.cpu(key+'-integration','scripts.audit_natural_chain_mixed_cpu',opts,target/'report.json','PASS_INDEPENDENT_MIXED_PILOT_INTEGRATION')
    def preflight(key):
        capture=native(key,'preflight');return key,capture,cpu.submit(integration,key,capture)
    def finish(key,parent,gate):
        checked=gate.result();execution.check();capture=native(key,'fit',parent);return key,capture,checked
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_COMPLETE_MIXED_FIRST_UPDATE_GATES',maximum_chips=16))
        basic=[k for k in plan['cases'] if k!='one-bridge-work']
        with cf.ThreadPoolExecutor(2) as workers:
            pending=[workers.submit(preflight,key) for key in basic]
            for future in cf.as_completed(pending):
                key,capture,gate=future.result();prefits[key]=(capture,gate)
        write_state(PUBLIC/'status.json',dict(status='RUNNING_BOUNDED_MIXED_CURRICULUM_FITS',maximum_chips=16))
        with cf.ThreadPoolExecutor(5) as workers:
            pending=[workers.submit(finish,key,*prefits[key]) for key in basic]
            parent,gate=prefits['one-bridge'];pending.append(workers.submit(finish,'one-bridge-work',parent,gate))
            for future in cf.as_completed(pending):
                key,capture,checked=future.result();fits[key]=capture;write_state(PUBLIC/'fit-progress.json',dict(completed={k:v[1] for k,v in fits.items()}))
        # All scientific fits are complete before any pilot quality readout.
        readout=plan['readout'];eval_cases={key:dict(arm=plan['cases'][key]['arm'],checkpoint='runs/'+capture[0]+'/'+capture[1]['checkpoint_file'],checkpoint_sha256=capture[1]['checkpoint_sha256'],depths=[1] if plan['cases'][key]['arm']=='one' else [1,2,4]) for key,capture in fits.items()}
        names={IDENTITY+'-readout-'+key:plan['root_comm_ports'][IDENTITY+'-readout-'+key] for key in fits};eval_plan={**plan,**readout,'identity':IDENTITY+'-readout','purpose':'readonly_original_chain_baseline','cases':eval_cases,'root_comm_ports':names};eval_file=out/'evaluation-plan.json';write_state(eval_file,eval_plan);write_state(PUBLIC/'evaluation-plan.json',eval_plan);eval_sha=file_sha256(eval_file)
        def read(key):
            name=IDENTITY+'-readout-'+key;opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/evaluation-plan.json','--plan-sha',eval_sha,'--case',key,'--comm-port',str(names[name])]
            native_row=execution.native(name,'scripts.read_natural_chain_native',opts,'COMPLETE_READONLY_ORIGINAL_CHAIN_BASELINE')
            target=out/'cpu'/'answers'/key;opts=['--root',str(root),'--plan',str(eval_file),'--plan-sha',eval_sha,'--case',key,'--capture',str(root/'runs'/name),'--output',str(target)]
            independent=cpu.submit(execution.cpu,key+'-answers','scripts.audit_natural_chain_readout_cpu',opts,target/'report.json','PASS_INDEPENDENT_ORIGINAL_CHAIN_READOUT')
            retention=out/'cpu'/'retention'/key;capture=fits[key];opts=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,'--case',key,'--checkpoint',str(root/'runs'/capture[0]/capture[1]['checkpoint_file']),'--checkpoint-sha',capture[1]['checkpoint_sha256'],'--output',str(retention)]
            natural=cpu.submit(execution.cpu,key+'-retention','scripts.score_natural_chain_retention_cpu',opts,retention/'report.json','COMPLETE_CHAIN_PILOT_RETENTION')
            return key,native_row,independent,natural
        write_state(PUBLIC/'status.json',dict(status='RUNNING_FIXED_PILOT_READOUTS',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as workers:
            for future in cf.as_completed([workers.submit(read,key) for key in fits]):
                key,row,independent,natural=future.result();records.append(dict(case=key,fit=fits[key][1],answers=row,independent_answers=independent.result(),retention=natural.result()));write_state(PUBLIC/'progress.json',dict(records=records))
        execution.check();analysis=analyze(root,records);write_state(PUBLIC/'analysis.json',analysis)
        result=dict(status='COMPLETE_NATURAL_CHAIN_CONTINUATION_PILOT',records=records,analysis=analysis,integration={k:gate.result() for k,(_,gate) in prefits.items()})
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records,completed_fits={k:v[1] for k,v in fits.items()});raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for sig,handler in handlers.items():signal.signal(sig,handler)
        result.update(plan_sha256=execution.sha,terminal_success=False,test_accessed=False,terminal_alpha_spent=0,scope=plan['scope']);write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();plan=register(a.root)
    if not a.prepare_only:run(a.root,plan)


if __name__=='__main__':main()
