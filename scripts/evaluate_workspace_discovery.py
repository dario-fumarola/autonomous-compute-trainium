"""Prefix-only held-out proposals; outcomes never select a workspace."""
import argparse
import json
import os
from pathlib import Path
import time
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.training import file_sha256
    from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig
    from compute_trainium.workspace_native_state import restore_model_shards
    from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor,drain_runtime
    from compute_trainium.workspace_learning import WorkspaceFormat,identity_reader_context
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_four_arm_discovery_pilot' or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Frozen held-out registration required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    # Every condition must pass its mature independent audit before ANY scores.
    for arm in plan['updates']:
        audit=json.loads((a.root/('workspace-discovery-audit-'+arm+'-v1/report.json')).read_text())
        if audit['status']!='PASS_WORKSPACE_DISCOVERY_MATURE_CPU_AUDIT':raise AssertionError('Unqualified condition cannot be scored')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    c0=torch.load(a.root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model'];config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config).eval()
    model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
    model.requires_grad_(False);model.compile_stages('neuron');actor=SelectiveWorkspacePredictor(model)
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']));data=WorkspaceViewBatches(a.root/plan['data_relative'],'tune')
    all_results={}
    with torch.no_grad():
        for arm in ['c0','workspace','ordinary','pause','draft']:
            if arm!='c0':
                path=a.root/'runs'/(plan['identity']+'-'+arm)/f'rank{rank}-u{plan["updates"][arm]}.zip'
                restore_model_shards(path,model,label='model');drain_runtime(actor.device)
            rows=[];actor.events=[]
            for batch in range(plan['evaluation_anchors']//world):
                index=batch*world+rank;view=data.at(index);c,y=view['prefix'],view['continuation']
                if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage='target_blind_evaluation',arm=arm,batch=batch,wall_seconds=time.time()-started)))
                # One draw, with a precommitted per-anchor seed. No reward-based
                # resampling, filtering, best-of-G, or answer-aware selection.
                z=None
                if arm in ['c0','workspace','draft']:
                    z=actor.sample(c,format=fmt,length=8,group_size=1,generator=torch.Generator().manual_seed(plan['evaluation_seed']+index))
                    # Reuse the qualified sum collective. Each integer-token
                    # row has exactly one contributing rank (IDs <2**24).
                    local=torch.zeros((world,8),dtype=torch.float32);local[rank]=z[0]
                    gathered=local.to('neuron');dist.all_reduce(gathered)
                    donor_rank=plan['evaluation_donors'][index]%world
                    donor=gathered.cpu()[donor_rank:donor_rank+1].long()
                raw32=actor.log_probs(c,y,'evaluation_raw_32').detach().cpu();drain_runtime(actor.device)
                row=dict(index=index,domain=view['domain'],source_row=view['source_row'],raw_nll=float(-raw32.mean()),raw_first4_nll=float(-raw32[:,:4].mean()),raw_first_token_nll=float(-raw32[:,0].mean()))
                row['primary_first_token_nll']=row['raw_first_token_nll']
                if z is not None:
                    for label,content in [('own',z),('donor',donor),('constant',z.new_full(z.shape,plan['pause_token_id'])),('privileged_copy_diagnostic_only',y[:,:8])]:
                        context=identity_reader_context(c,content,fmt)
                        value=actor.log_probs(context,y[:,:4],'evaluation_'+label).detach().cpu();drain_runtime(actor.device)
                        row[label+'_nll']=float(-value.mean());row[label+'_first_token_nll']=float(-value[:,0].mean())
                    row['primary_nll']=row['raw_first4_nll'] if arm=='c0' else row['own_nll']
                    if arm!='c0':row['primary_first_token_nll']=row['own_first_token_nll']
                elif arm=='pause':
                    context=torch.cat([c,c.new_full((1,8),plan['pause_token_id'])],1)
                    value=actor.log_probs(context,y[:,:4],'evaluation_trained_pause').detach().cpu();drain_runtime(actor.device)
                    row['primary_nll']=float(-value.mean())
                    row['primary_first_token_nll']=float(-value[:,0].mean())
                else:row['primary_nll']=row['raw_first4_nll']
                rows.append(row)
            (out/f'rank{rank}-{arm}-losses.json').write_text(json.dumps(rows,indent=2)+'\n')
            dist.barrier()
            if rank==0:
                all_rows=[x for k in range(world) for x in json.loads((out/f'rank{k}-{arm}-losses.json').read_text())]
                if sorted(x['index'] for x in all_rows)!=list(range(plan['evaluation_anchors'])):raise AssertionError('Exact held-out coverage required')
                fields=[k for k in all_rows[0] if k.endswith('_nll')]
                scores={field:sum(weight*sum(x[field] for x in all_rows if x['domain']==d)/sum(x['domain']==d for x in all_rows) for d,weight in enumerate([.6,.2,.2])) for field in fields}
                all_results[arm]=dict(scores=scores,anchors=len(all_rows),ledger=actor.ledger())
                (out/'aggregate-progress.json').write_text(json.dumps(all_results,indent=2)+'\n')
    fallback=torch_neuronx.get_fallback_ops()
    if fallback:raise AssertionError('Held-out scoring used CPU fallback:'+str(fallback))
    if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_WORKSPACE_DISCOVERY_TARGET_BLIND_EVALUATION',world=world,
        models=all_results,wall_seconds=time.time()-started,fallback_ops=fallback,optimizer_updates=0,
        warning='320 TUNE anchors from reused source groups;single seed,constant short pilot,not confirmation or reasoning evidence.Base pretraining contamination cannot be excluded.All content interventions recompute full states;privileged future copy is a diagnostic only.Not deployed thought selection.'),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
