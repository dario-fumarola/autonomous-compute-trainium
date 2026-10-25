"""Recover qualified scientific foundation bodies and unchanged task orders."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.natural_chain_mixed import update_matrix_work
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest

IDENTITY='natural-chain-foundation-preparation-v1'
PUBLIC=Path('results/natural_chain_foundation_preparation_20261025')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time();root=a.root;prior=root/'natural-chain-continuation-pilot-v2';old=json.loads((prior/'plan.json').read_text())
    if json.loads((prior/'report.json').read_text())['status']!='COMPLETE_NATURAL_CHAIN_CONTINUATION_PILOT':raise ValueError('Completed original fixed pilot required')
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    sources=['scripts/prepare_natural_chain_foundations.py','docs/NATURAL_CHAIN_FOUNDATION_PREPARATION_20261025.md','compute_trainium/natural_chain_mixed.py','compute_trainium/parcae_primitive_sdpa_candidate.py']
    folders=dict(one='future-span-work-control-v1-one-ntp-s5231-lr32000-work-finish',recurrent='future-span-main-v2-recurrent-ntp-s5231-lr32000-finish')
    plan=dict(identity=IDENTITY,registered_unix=started,source_sha256={p:file_sha256(p) for p in sources},parent_pilot_plan_sha256=file_sha256(prior/'plan.json'),parent_pilot_report_sha256=file_sha256(prior/'report.json'),source_tasks_sha256=old['tasks_sha256'],parent_folders=folders,seconds_limit=600,scope=Path(sources[1]).read_text(),assigned_chips=0,optimizer_updates=0,scientific_launch_allowed=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);result=dict(status='STOPPED_FOUNDATION_BODY_PREPARATION');bodies={};initial_shas=set();packing=set()
    def guard():
        require_inode_reserve([root,Path('/')]);free=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if time.time()-started>600 or free<512*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Foundation CPU preparation cap/reserve')
    try:
        torch.set_num_threads(8);config=ParcaeCandidateConfig(**old['model']);optimizer_ops=None
        for role,folder in folders.items():
            guard();base=root/'runs'/folder;report=json.loads((base/'report.json').read_text());record=report['checkpoint_records'][-1];file=base/f"checkpoint-u{record['update']}.pt"
            if file_sha256(file)!=record['checkpoint_sha256']:raise AssertionError('Original foundation state changed')
            saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
            if saved['qualification_only'] or sig['seed']!=5231 or sig['arm']!=role or sig['objective_weight']!=0 or saved['cursor']!=record['update']*512:raise ValueError('Original scientific role/history differs')
            state_digest=digest(dict(model=dict(sorted(saved['model'].items())),optimizer=saved['optimizer']))
            if state_digest!=record['full_state_digest'] or state_digest!=saved['model_optimizer_digest']:raise AssertionError('Complete original model/optimizer digest differs')
            initial_shas.add(sig['initial_sha256']);packing.add((sig['packing_fingerprint'],sig['corpus_sha256']));model=ParcaePrimitiveSDPACandidate(config,recurrent=True,initialize=False,fp16_body=True)
            model.load_state_dict({k:v for k,v in saved['model'].items() if not k.startswith('future_lm_head.') and k!='future_positions'},strict=True);body_digest=digest(dict(sorted(model.state_dict().items())))
            destination=out/(role+'-body.pt');atomic_checkpoint(destination,dict(model=model.state_dict(),purpose='natural_chain_foundation_initializer',qualification_only=False,role=role,parent_checkpoint_sha256=record['checkpoint_sha256'],parent_full_state_digest=state_digest,parent_targets=saved['real_targets'],parent_cursor=saved['cursor'],optimizer_reset_required=True))
            restored=torch.load(destination,map_location='cpu',mmap=True,weights_only=False)
            if digest(dict(sorted(restored['model'].items())))!=body_digest:raise AssertionError('Prediction body readback differs')
            count=sum(p.numel() for p in model.parameters());current=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
            if optimizer_ops is not None and current!=optimizer_ops:raise AssertionError('Foundation optimizer parameter inventory differs')
            optimizer_ops=current;bodies[role]=dict(file=destination.name,sha256=file_sha256(destination),model_digest=body_digest,parent_checkpoint_sha256=record['checkpoint_sha256'],parent_full_state_digest=state_digest,parent_real_targets=saved['real_targets'],parent_cursor=saved['cursor'],parent_nominal_matrix_ops=saved['estimated_dense_matrix_ops'],parameters=count,original_initializer_sha256=sig['initial_sha256'])
            del model,saved,restored
        if len(initial_shas)!=1 or len(packing)!=1 or bodies['one']['model_digest']==bodies['recurrent']['model_digest']:raise AssertionError('Distinct scientific paths must share the original initializer/source contract')
        r_budget=bodies['recurrent']['parent_nominal_matrix_ops']+old['cases']['recurrent-bridge']['expected_total_matrix_ops'];one_base=bodies['one']['parent_nominal_matrix_ops'];per_update=update_matrix_work(config,1,task_weight=.3,optimizer_per_replica=optimizer_ops);long_updates=(r_budget-one_base)//per_update
        if long_updates!=245:raise AssertionError('Unexpected complete-history work endpoint;reconcile before admission')
        task_file=prior/'tasks.pt'
        if file_sha256(task_file)!=old['tasks_sha256']:raise AssertionError('Original task sampler changed')
        tasks=torch.load(task_file,map_location='cpu',mmap=True,weights_only=False);generator=torch.Generator(device='cpu').manual_seed(old['sampling_seed']);flat=[]
        while len(flat)<long_updates*128:flat.extend(torch.randperm(old['unique_problems'],generator=generator).tolist())
        extended=torch.tensor(flat[:long_updates*128],dtype=torch.long).view(long_updates,128)
        if not torch.equal(extended[:len(tasks['order'])],tasks['order']):raise AssertionError('Existing task order prefix changed')
        tasks['order']=extended;atomic_checkpoint(out/'tasks.pt',tasks)
        manifest=dict(identity=IDENTITY,plan_sha256=file_sha256(out/'plan.json'),model=old['model'],initials=bodies,data=old['data'],ntp_start_cursor=max(v['parent_cursor'] for v in bodies.values()),tasks_sha256=file_sha256(out/'tasks.pt'),source_views_sha256=old['source_views_sha256'],noise_seed=old['noise_seed'],sampling_seed=old['sampling_seed'],unique_problems=old['unique_problems'],optimizer_matrix_ops_per_replica=optimizer_ops,
            complete_history_work_control=dict(updates=long_updates,r_budget=r_budget,one_history_plus_updates=one_base+long_updates*per_update,relative_shortfall=(r_budget-one_base-long_updates*per_update)/r_budget,next_update_would_exceed_budget=one_base+(long_updates+1)*per_update>r_budget),scientific_launch_allowed=False)
        guard()
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):raise AssertionError('Frozen foundation preparation source changed')
        write_state(out/'manifest.json',manifest);write_state(PUBLIC/'manifest.json',manifest);result=dict(status='COMPLETE_EXISTING_FOUNDATION_BODY_PREPARATION',manifest_sha256=file_sha256(out/'manifest.json'),same_original_random_initializer=True,distinct_body_weights=True,full_parent_state_checks_pass=True,body_readbacks_exact=True,task_order_prefix_exact=True,common_future_ntp_cursor=manifest['ntp_start_cursor'],matched_one_curriculum_updates=long_updates,scientific_launch_allowed=False)
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0,wall_seconds=time.time()-started);write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
