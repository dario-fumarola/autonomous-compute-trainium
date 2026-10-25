"""One bounded retained-foundation comparison with source/work controls."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_natural_chain_pilot as original

IDENTITY='natural-chain-foundation-pilot-v1'
PUBLIC=Path('results/natural_chain_foundation_pilot_20261025')
BASE_ANALYZE=original.analyze

class FoundationExecution(Execution):
    def check(self):super().check();require_inode_reserve([self.root,Path('/')])
    def native(self,identity,module,options,status):
        if module=='scripts.train_natural_chain_pilot':module='scripts.train_natural_chain_foundation'
        return super().native(identity,module,options,status)
    def cpu(self,identity,module,options,report_file,status):
        if module=='scripts.audit_natural_chain_mixed_cpu':module='scripts.audit_natural_chain_foundation_cpu'
        return super().cpu(identity,module,options,report_file,status)


def register(root):
    import torch
    from compute_trainium.natural_chain_foundation_state import foundation_initial
    from compute_trainium.natural_chain_mixed import update_matrix_work
    from compute_trainium.natural_chain_objective import task_depth
    from compute_trainium.natural_chain_pilot_data import answer_count
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.training import PackedCorpus,atomic_checkpoint
    from scripts.qualify_native import digest
    torch.set_num_threads(8);require_inode_reserve([root,Path('/')])
    if managed_running():raise RuntimeError('All predecessor native owners must be terminal')
    previous=root/'natural-chain-continuation-pilot-v2';prep=root/'natural-chain-foundation-preparation-v1';manifest_file=prep/'manifest.json';ready=json.loads((prep/'report.json').read_text())
    if ready['status']!='COMPLETE_EXISTING_FOUNDATION_BODY_PREPARATION' or file_sha256(manifest_file)!=ready['manifest_sha256']:raise ValueError('Verified existing foundation preparation required')
    m=json.loads(manifest_file.read_text());old=json.loads((previous/'plan.json').read_text());initials={role:{**row,'role':role} for role,row in m['initials'].items()}
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    for role,row in initials.items():
        asset=prep/row['file']
        if file_sha256(asset)!=row['sha256']:raise AssertionError('Verified foundation body changed')
        (out/row['file']).symlink_to(os.path.relpath(asset,out))
    # Add the original source-matched One as a separate control,never substitute
    # it for the stronger work-matched One primary comparator.
    folder=root/'runs/future-span-main-v2-one-ntp-s5231-lr32000-finish';report=json.loads((folder/'report.json').read_text());record=report['checkpoint_records'][-1];checkpoint=folder/f"checkpoint-u{record['update']}.pt"
    if file_sha256(checkpoint)!=record['checkpoint_sha256']:raise AssertionError('Original source-matched One changed')
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
    if saved['qualification_only'] or sig['arm']!='one' or sig['seed']!=5231 or sig['objective_weight']!=0 or sig['initial_sha256']!=initials['one']['original_initializer_sha256'] or sig['packing_fingerprint']!=m['data']['packing_fingerprint']:raise ValueError('Source-matched One history differs')
    full=digest(dict(model=dict(sorted(saved['model'].items())),optimizer=saved['optimizer']))
    if full!=record['full_state_digest'] or saved['real_targets']!=initials['recurrent']['parent_real_targets'] or saved['cursor']!=initials['recurrent']['parent_cursor']:raise AssertionError('Original same-source state/digest differs')
    cfg=ParcaeCandidateConfig(**m['model']);model=ParcaePrimitiveSDPACandidate(cfg,recurrent=True,initialize=False,fp16_body=True);model.load_state_dict({k:v for k,v in saved['model'].items() if not k.startswith('future_lm_head.') and k!='future_positions'},strict=True)
    asset=out/'one-source-body.pt';body_digest=digest(dict(sorted(model.state_dict().items())))
    atomic_checkpoint(asset,dict(model=model.state_dict(),purpose='natural_chain_foundation_initializer',qualification_only=False,role='one',parent_checkpoint_sha256=record['checkpoint_sha256'],parent_full_state_digest=full,parent_targets=saved['real_targets'],parent_cursor=saved['cursor'],optimizer_reset_required=True))
    if digest(dict(sorted(torch.load(asset,map_location='cpu',mmap=True,weights_only=False)['model'].items())))!=body_digest:raise AssertionError('Source One body readback differs')
    initials['one_source']=dict(file=asset.name,sha256=file_sha256(asset),model_digest=body_digest,role='one',parent_checkpoint_sha256=record['checkpoint_sha256'],parent_full_state_digest=full,parent_real_targets=saved['real_targets'],parent_cursor=saved['cursor'],parent_nominal_matrix_ops=saved['estimated_dense_matrix_ops'],parameters=sum(p.numel() for p in model.parameters()),original_initializer_sha256=sig['initial_sha256']);del model,saved
    if len({v['parameters'] for v in initials.values()})!=1 or len({v['original_initializer_sha256'] for v in initials.values()})!=1:raise AssertionError('Original initializer or parameter matching fails')
    task_file=prep/'tasks.pt'
    if file_sha256(task_file)!=m['tasks_sha256']:raise AssertionError('Frozen unchanged task sampler differs')
    (out/'tasks.pt').symlink_to(os.path.relpath(task_file,out));tasks=torch.load(task_file,map_location='cpu',mmap=True,weights_only=False)
    cases={};optimizer_ops=m['optimizer_matrix_ops_per_replica']
    for key,arm,initial,weight in [('one-bridge','one','one',.3),('recurrent-bridge','recurrent','recurrent',.3),('one-ntp','one','one',0.),('recurrent-ntp','recurrent','recurrent',0.),('one-source-bridge','one','one_source',.3),('one-source-ntp','one','one_source',0.)]:
        work=sum(update_matrix_work(cfg,task_depth(arm,u),task_weight=weight,optimizer_per_replica=optimizer_ops) for u in range(200));cases[key]=dict(arm=arm,initial=initial,task_weight=weight,total_updates=200,peak_lr=.001,expected_total_matrix_ops=work,retained_history_plus_adaptation_matrix_ops=initials[initial]['parent_nominal_matrix_ops']+work)
    work_control=m['complete_history_work_control'];count=work_control['updates'];one_work=count*update_matrix_work(cfg,1,task_weight=.3,optimizer_per_replica=optimizer_ops)
    cases['one-bridge-work']=dict(arm='one',initial='one',task_weight=.3,total_updates=count,peak_lr=.001,parent_case='one-bridge',expected_total_matrix_ops=one_work,retained_history_plus_adaptation_matrix_ops=initials['one']['parent_nominal_matrix_ops']+one_work,relative_work_shortfall=work_control['relative_shortfall'],next_update_would_exceed_budget=work_control['next_update_would_exceed_budget'])
    if cases['recurrent-bridge']['retained_history_plus_adaptation_matrix_ops']!=work_control['r_budget'] or cases['one-bridge-work']['retained_history_plus_adaptation_matrix_ops']!=work_control['one_history_plus_updates']:raise AssertionError('Retained-history work control changed')
    packed=PackedCorpus.from_mapped(root/m['data']['packed'],1024,23,m['data']['corpus_sha256']);start=m['ntp_start_cursor']
    if start!=max(v['parent_cursor'] for v in initials.values()) or start+count*128>packed.windows:raise ValueError('Common unseen source cohort unavailable')
    for key,case in cases.items():case.update(expected_ntp_targets=sum(packed.counts(start,case['total_updates']*128)),expected_answer_targets=sum(answer_count(tasks,u,case) for u in range(case['total_updates'])) if case['task_weight'] else 0)
    del tasks,packed
    extras=['scripts/run_natural_chain_foundation.py','scripts/train_natural_chain_foundation.py','scripts/audit_natural_chain_foundation_cpu.py','compute_trainium/natural_chain_foundation_state.py','tests/test_natural_chain_foundation_state.py','docs/NATURAL_CHAIN_FOUNDATION_PILOT_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in extras}}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise AssertionError('Original mixed implementation or new parent-selection source changed')
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-'+key+'-'+mode for key in cases for mode in (['fit'] if key=='one-bridge-work' else ['preflight','fit'])]+[IDENTITY+'-readout-'+key for key in cases]
    plan=copy.deepcopy(old)
    for key in ['initial_sha256','initial_model_digest','parent_checkpoint_sha256','parent_targets','parent_nominal_matrix_ops']:plan.pop(key,None)
    plan.update(identity=IDENTITY,registered_unix=time.time(),purpose='natural_chain_foundation_history_pilot',source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,preparation=IDENTITY,
        initial_states=initials,prepared_manifest_sha256=file_sha256(manifest_file),cases=cases,tasks_sha256=m['tasks_sha256'],ntp_start_cursor=start,
        original_negative_pilot_report_sha256=file_sha256(previous/'report.json'),source_matched_whole_history_controls=['one-source-ntp','one-source-bridge'],
        primary_comparison='recurrent-bridge_depth2_minus_one-bridge-work_depth1',primary_scope='domain_development_only_five_percentage_point_screen',
        seconds_limit=7200,phase_seconds_limit=1200,cpu_phase_seconds_limit=1200,root_comm_ports=reserve_ports(root,names),scope=Path(extras[-1]).read_text())
    for case in cases.values():foundation_initial(plan,case)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);return plan


def analyze(root,records):
    result=BASE_ANALYZE(root,records);effects=result['effects'];effects['equal_continuation_stronger_one_comparator']=effects.pop('primary_recurrence_over_same_curriculum');effects['primary_full_history_work_matched']=effects.pop('recurrence_over_longer_one_work_control')
    rows={key:{(r['document_id'],r['depth']):r for r in json.loads((root/'runs'/(IDENTITY+'-readout-'+key)/'records.json').read_text())} for key in ['recurrent-bridge','recurrent-ntp','one-source-bridge','one-source-ntp']};ids=sorted(k[0] for k in rows['one-source-bridge']);groups={}
    for i in ids:groups.setdefault(rows['one-source-bridge'][(i,1)]['group'],[]).append(i)
    def value(key,depth,i):return float(rows[key][(i,depth)]['correct'])
    functions=dict(source_matched_recurrence_advantage=lambda i:value('recurrent-bridge',2,i)-value('one-source-bridge',1,i),source_matched_objective_interaction=lambda i:(value('recurrent-bridge',2,i)-value('one-source-bridge',1,i))-(value('recurrent-ntp',2,i)-value('one-source-ntp',1,i)))
    for label,fn in functions.items():
        keys=sorted(groups);rng=random.Random(1327);per={g:[sum(fn(i) for i in members),len(members)] for g,members in groups.items()};draws=[]
        for _ in range(3000):
            chosen=rng.choices(keys,k=len(keys));draws.append(sum(per[g][0] for g in chosen)/sum(per[g][1] for g in chosen))
        draws.sort();effect=sum(fn(i) for i in ids)/len(ids);effects[label]=dict(accuracy_difference=effect,conditional_source_group_95_interval=[draws[74],draws[2924]],observed_at_least_five_percentage_points=effect>=.05)
    result.update(primary_effect='primary_full_history_work_matched',scope='Consumed discovery DEV,one original pretrained seed/one adaptation stream. Separate source-matched and retained-history nominal-work controls;different trained initial bodies are intentional. No independent confirmation,broad-goal completion or terminal alpha.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();plan=register(a.root)
    if a.prepare_only:return
    original.IDENTITY=IDENTITY;original.PUBLIC=PUBLIC;original.Execution=FoundationExecution;original.analyze=analyze;original.run(a.root,plan)
    file=a.root/IDENTITY/'report.json';d=json.loads(file.read_text());d.update(status='COMPLETE_NATURAL_CHAIN_FOUNDATION_HISTORY_PILOT');write_state(file,d);write_state(PUBLIC/'report.json',d);write_state(PUBLIC/'status.json',dict(status=d['status'],assigned_chips=0))


if __name__=='__main__':main()
