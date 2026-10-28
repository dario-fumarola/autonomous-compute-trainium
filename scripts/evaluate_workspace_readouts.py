"""Frozen five-model2x2 readout diagnostic;zero learning or answer selection."""
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
    from compute_trainium.workspace_learning import WorkspaceFormat
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    from compute_trainium.workspace_readout_diagnostic import readout_context,VARIANTS
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_readout_diagnostic' or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Frozen diagnostic required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    for files in plan['retained_endpoint_proofs'].values():
        if any(file_sha256(a.root/n)!=h for n,h in files.items()):raise AssertionError('Qualified endpoints changed')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    c0=torch.load(a.root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model'];config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config).eval()
    model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
    model.requires_grad_(False);model.compile_stages('neuron');actor=SelectiveWorkspacePredictor(model)
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']));data=WorkspaceViewBatches(a.root/plan['data_relative'],'tune');all_results={};private=[];parity=0.
    def score(context,targets,label):
        p=actor.log_probs(context,targets,label).detach().cpu();drain_runtime(actor.device)
        return float(-p.mean()),float(-p[:,0].mean())
    with torch.no_grad():
        for arm in ['c0','workspace','ordinary','pause','draft']:
            if arm!='c0':
                path=a.root/'runs'/plan['arm_runs'][arm]/f'rank{rank}-u{plan["updates"][arm]}.zip'
                restore_model_shards(path,model,label='model');drain_runtime(actor.device)
            original=json.loads((a.root/'runs'/plan['original_evaluation_run']/f'rank{rank}-{arm}-losses.json').read_text());old={r['index']:r for r in original}
            rows=[];actor.events=[]
            for batch in range(5):
                index=batch*world+rank;view=data.at(index);c,y=view['prefix'],view['continuation']
                if view['source_row']!=old[index]['source_row'] or view['domain']!=old[index]['domain']:raise AssertionError('Original source order changed')
                if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage='readout_diagnostic',arm=arm,batch=batch,wall_seconds=time.time()-started)))
                p=actor.log_probs(c,y,'raw32').detach().cpu();drain_runtime(actor.device)
                row=dict(index=index,domain=view['domain'],source_row=view['source_row'],raw_nll=float(-p.mean()),raw_first4_nll=float(-p[:,:4].mean()),raw_first_token_nll=float(-p[:,0].mean()))
                for k in ['raw_nll','raw_first4_nll','raw_first_token_nll']:parity=max(parity,abs(row[k]-old[index][k]))
                value,first=score(torch.cat([c,c[:,-16:]],1),y[:,:4],'tail_only')
                row.update(tail_only_nll=value,tail_only_first_token_nll=first)
                for style in ['sample','greedy']:
                    z=actor.sample(c,format=fmt,length=8,group_size=1,generator=torch.Generator().manual_seed(plan['evaluation_seed']+index),greedy=style=='greedy')
                    local=torch.zeros((world,8),dtype=torch.float32);local[rank]=z[0];gathered=local.to('neuron');dist.all_reduce(gathered)
                    k=plan['evaluation_donors'][index]%world;donor=gathered.cpu()[k:k+1].long()
                    private.append(dict(arm=arm,index=index,style=style,workspace=z))
                    for tail in [False,True]:
                        variant=style+('_tail' if tail else '_prepend')
                        for label,content in [('own',z),('donor',donor),('constant',z.new_full(z.shape,plan['pause_token_id']))]:
                            value,first=score(readout_context(c,content,fmt,tail=tail),y[:,:4],variant+'_'+label)
                            row[variant+'.'+label+'_nll']=value;row[variant+'.'+label+'_first_token_nll']=first
                        if variant=='sample_prepend' and 'own_nll' in old[index]:
                            parity=max(parity,abs(row[variant+'.own_nll']-old[index]['own_nll']),abs(row[variant+'.own_first_token_nll']-old[index]['own_first_token_nll']))
                for tail in [False,True]:
                    label='copy_tail' if tail else 'copy_prepend'
                    value,first=score(readout_context(c,y[:,:8],fmt,tail=tail),y[:,:4],label+'_diagnostic_only')
                    row[label+'_nll']=value;row[label+'_first_token_nll']=first
                if parity>1e-5:raise AssertionError('Untouched raw/original sampled outputs changed')
                rows.append(row)
            (out/f'rank{rank}-{arm}-losses.json').write_text(json.dumps(rows,indent=2)+'\n');dist.barrier()
            if rank==0:
                all_rows=[x for r in range(world) for x in json.loads((out/f'rank{r}-{arm}-losses.json').read_text())]
                if sorted(x['index'] for x in all_rows)!=list(range(320)):raise AssertionError('Exact original panel required')
                fields=[k for k in all_rows[0] if k.endswith('_nll')]
                scores={field:sum(w*sum(x[field] for x in all_rows if x['domain']==d)/sum(x['domain']==d for x in all_rows) for d,w in enumerate([.6,.2,.2])) for field in fields}
                all_results[arm]=dict(scores=scores,anchors=320,ledger=actor.ledger());(out/'aggregate-progress.json').write_text(json.dumps(all_results,indent=2)+'\n')
    torch.save(private,out/f'rank{rank}-PRIVATE-generated-notes.pt')
    errors=torch.zeros(world,dtype=torch.float32);errors[rank]=parity;errors=errors.to('neuron');dist.all_reduce(errors);parity=float(errors.cpu().max())
    fallback=torch_neuronx.get_fallback_ops()
    if fallback:raise AssertionError('Diagnostic CPU fallback:'+str(fallback))
    if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_WORKSPACE_READOUT_DIAGNOSTIC',world=world,models=all_results,optimizer_updates=0,
        original_raw_and_sample_parity_max_abs_error=parity,wall_seconds=time.time()-started,fallback_ops=fallback,
        warning='Outcome-informed2x2 readout development diagnostic on same320 TUNE anchors.All models receive identical inference choices.No training,selection by answer,fresh confirmation,allocation or transferable reasoning claim.Replayed tail is from prefix only;copy control is privileged diagnostic only.'),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
