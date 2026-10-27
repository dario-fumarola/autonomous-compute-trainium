"""Synthetic numerical arrays only:sharded OLMo/AdamW/EMA and fresh-process replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def worker(rank,world,backend,output,mode,parent=None,port=None):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
    from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator,sharded_ema
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_sharded_candidate import local_parameter
    from compute_trainium.scale_sharded_fixture import cpu_tree,relative
    from scripts.qualify_native import digest
    from torch.distributed.tensor import DTensor
    started=time.time();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if backend=='neuron':
        os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(port)
        import torch_neuronx
        torch.neuron.set_device(0);device='neuron';torch.set_num_threads(2)
    else:device='cpu';torch.set_num_threads(1)
    dist.init_process_group(backend,rank=rank,world_size=world);torch.manual_seed(7107)
    config=OlmoWorkspaceConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=4,vocab_size=128,pad_token_id=127)
    reference=WorkspaceOlmoCandidate(config).eval();ema_reference=copy.deepcopy(reference)
    reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
    initial={n:p.detach().clone() for n,p in reference.state_dict().items()};mesh=init_device_mesh(device,(world,))
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
    model=shard_model_from_cpu(model,mesh,initial,device)
    with torch.device('meta'):scorer=WorkspaceOlmoCandidate(config).eval()
    scorer=shard_model_from_cpu(scorer,mesh,initial,device).requires_grad_(False)
    optimizer=WorkspaceShardedAdamW(model,lr=1e-4,fused=backend=='neuron')
    if backend=='neuron':model.compile_stages('neuron')
    cos,sin=rotary_tensors(config,8,device);cpu_cos,cpu_sin=rotary_tensors(config,8)
    if mode=='replay':
        saved=torch.load(Path(parent)/f'rank{rank}-u2.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():
            for label,current in [('model',model),('scorer',scorer)]:
                for n,p in current.named_parameters():local_parameter(p).copy_(saved[label][n].to(device))
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        cpu=torch.load(Path(parent)/'reference-u2.pt',weights_only=True)
        reference.load_state_dict(cpu['model']);ema_reference.load_state_dict(cpu['scorer']);reference_optimizer.load_state_dict(cpu['optimizer'])
    rows=[]
    for update in ([3] if mode=='replay' else [1,2,3]):
        inputs=(torch.arange(world*8).reshape(world,8)+update*3)%126+1;targets=(inputs+2)%126+1
        optimizer.zero_grad();reference_optimizer.zero_grad(set_to_none=True)
        before={n:p.full_tensor().detach().cpu() for n,p in model.named_parameters()}
        cpu_before={n:p.detach().clone() for n,p in reference.named_parameters()}
        ema_before={n:p.full_tensor().detach().cpu() for n,p in scorer.named_parameters()}
        cpu_ema_before={n:p.detach().clone() for n,p in ema_reference.named_parameters()}
        accumulator=ShardedGradientAccumulator(model)
        for _ in range(2):
            logits=model(inputs[rank:rank+1].to(device),cos,sin)
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,128),targets[rank].to(device).reshape(-1))/world
            (loss*128).backward();accumulator.add_and_clear()
            assert all(p.grad is None for p in model.parameters())
        accumulator.install();norm=optimizer.finish_gradients(scale=256)
        expected=reference(inputs,cpu_cos,cpu_sin)
        cpu_loss=torch.nn.functional.cross_entropy(expected.reshape(-1,128),targets.reshape(-1));cpu_loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
        gradients={n:p.grad.full_tensor().cpu() for n,p in model.named_parameters()}
        comparison=relative({n:p.grad for n,p in reference.named_parameters()},gradients)
        optimizer.step();reference_optimizer.step();sharded_ema(scorer,model,decay=.5)
        with torch.no_grad():
            for a,b in zip(ema_reference.parameters(),reference.parameters(),strict=True):a.mul_(.5).add_(b,alpha=.5)
        weights={n:p.full_tensor().cpu() for n,p in model.named_parameters()}
        ema={n:p.full_tensor().cpu() for n,p in scorer.named_parameters()}
        changes=relative(reference.state_dict(),weights);ema_error=relative(ema_reference.state_dict(),ema)
        delta=relative({n:p.detach()-cpu_before[n] for n,p in reference.named_parameters()},
                       {n:weights[n]-before[n] for n in weights})
        actual_moments={};expected_moments={};cpu_params=dict(reference.named_parameters())
        for local,(name,p) in zip(optimizer.local,optimizer.named,strict=True):
            for field in ['exp_avg','exp_avg_sq']:
                value=optimizer.adam.state[local][field]
                full=DTensor.from_local(value,mesh,p.placements,shape=p.shape,stride=p.stride(),run_check=False).full_tensor().cpu()
                actual_moments[name+'.'+field]=full
                expected_moments[name+'.'+field]=reference_optimizer.state[cpu_params[name]][field]
        moments=relative(expected_moments,actual_moments)
        ema_delta=relative({n:p.detach()-cpu_ema_before[n] for n,p in ema_reference.named_parameters()},
                          {n:ema[n]-ema_before[n] for n in ema})
        row=dict(update=update,gradient=comparison,weights=changes,parameter_delta=delta,moments=moments,
                 ema=ema_error,ema_delta=ema_delta,gradient_norm=norm)
        if max(v['largest_named_relative_l2'] for v in [comparison,delta,moments,ema_delta])>.02:raise AssertionError('Complete independent CPU gradient/update/moment/EMA fixture mismatch')
        if update==2:
            from compute_trainium.workspace_native_state import save_complete_state,restore_complete_state,restore_model_shards
            generator=torch.Generator().manual_seed(7289+rank)
            saved_group=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.adam.param_groups]
            save_complete_state(output/f'rank{rank}-complete.zip',model,scorer,optimizer,generator,extra={'fixture':True})
            state_before={n:local_parameter(p).detach().clone() for n,p in model.named_parameters()}
            for alias in optimizer.local:optimizer.adam.state[alias].clear()
            restore_complete_state(output/f'rank{rank}-complete.zip',model,scorer,optimizer,generator)
            assert all(torch.equal(local_parameter(p),state_before[n]) for n,p in model.named_parameters())
            assert saved_group==[{k:v for k,v in group.items() if k!='params'} for group in optimizer.adam.param_groups]
            scorer_before={n:local_parameter(p).detach().clone() for n,p in scorer.named_parameters()}
            with torch.no_grad():
                for current in [model,scorer]:
                    for p in current.parameters():local_parameter(p).reshape(-1)[0].add_(1.)
            restore_model_shards(output/f'rank{rank}-complete.zip',model,label='model')
            restore_model_shards(output/f'rank{rank}-complete.zip',scorer,label='scorer')
            assert all(torch.equal(local_parameter(p),state_before[n]) for n,p in model.named_parameters())
            assert all(torch.equal(local_parameter(p),scorer_before[n]) for n,p in scorer.named_parameters())
            torch.save(cpu_tree(dict(model={n:local_parameter(p) for n,p in model.named_parameters()},
                scorer={n:local_parameter(p) for n,p in scorer.named_parameters()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state())),output/f'rank{rank}-u2.pt')
            if rank==0:torch.save(dict(model=reference.state_dict(),scorer=ema_reference.state_dict(),optimizer=reference_optimizer.state_dict()),output/'reference-u2.pt')
        if update==3:
            if mode=='replay':
                previous=torch.load(Path(parent)/f'expected-rank{rank}-u3.pt',weights_only=True)
                row['fresh_process_exact_replay']=(all(torch.equal(weights[n],v) for n,v in previous['model'].items())
                    and all(torch.equal(ema[n],v) for n,v in previous['scorer'].items())
                    and digest(cpu_tree(optimizer.state_dict()))==digest(previous['optimizer'])
                    and torch.equal(torch.get_rng_state(),previous['rng']))
                if not row['fresh_process_exact_replay']:raise AssertionError('Fresh-process exact weights/EMA replay differs')
            else:torch.save(dict(model=weights,scorer=ema,optimizer=cpu_tree(optimizer.state_dict()),rng=torch.get_rng_state()),output/f'expected-rank{rank}-u3.pt')
        rows.append(row);dist.barrier()
    if rank==0:
        (output/'report.json').write_text(json.dumps(dict(status='PASS_WORKSPACE_SHARDED_SOFTWARE_FIXTURE',
            mode=mode,world=world,records=rows,wall_seconds=time.time()-started,research_examples=0,
            research_optimizer_updates=0,actual_numerical_updates=len(rows)),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--corpus')
    p.add_argument('--mode',choices=['warm','replay'],required=True);p.add_argument('--parent',type=Path);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    a=p.parse_args()
    from compute_trainium.training import file_sha256
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if (file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_native_sharded_software_fixture'
            or plan['world']!=64 or plan['phases'][a.output.name]['mode']!=a.mode
            or plan['root_comm_ports'][a.output.name]!=a.comm_port):raise ValueError('Exact frozen fixture plan required')
    if any(file_sha256(source/name)!=sha for name,sha in plan['worker_source_sha256'].items()):raise AssertionError('Frozen fixture source changed')
    worker(int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),'neuron',a.output,a.mode,a.parent,a.comm_port)
