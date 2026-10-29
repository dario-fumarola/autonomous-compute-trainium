"""One disposable update from a retained checkpoint; no scientific fitting."""
import argparse
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main(a):
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.chunked_persistent_factory import make_model
    from compute_trainium.interleaved_objective import target_loss_sum
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.bounded_replica_capture import capture_replicas
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(source/f)!=h for f,h in p['worker_source_sha256'].items()):
        raise AssertionError('Frozen completion source changed')
    if p['purpose']!='interleaved_workspace_actual_shape_qualification' or p['scientific_fit_admitted'] or p['retained_for_science']:
        raise AssertionError('Disposable completion only')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=64:raise AssertionError('All64 ranks required')
    torch.set_num_threads(1);os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    torch.neuron.set_device(0);dist.init_process_group('neuron');start=time.time()
    a.output.mkdir(parents=True,exist_ok=True)
    parent=a.root/p['completion_checkpoint_relative']
    if file_sha256(parent)!=p['completion_checkpoint_sha256']:
        raise AssertionError('Original retained update103 changed')
    torch.manual_seed(p['seed']);model=make_model(p,explicit_attention=True).to('neuron')
    optimizer=torch.optim.AdamW(parameter_groups(model,model.embedding.weight),lr=p['lr'],
        betas=(.9,.95),eps=1e-8,foreach=False,fused=True)
    saved=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
    if saved['update']!=103:raise AssertionError('Exactly the retained pre-update104 state required')
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    expected=saved['digest']
    restored=digest(cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state())))
    if restored!=expected:raise AssertionError('Full model/optimizer/RNG restoration differs')
    (a.output/f'PRIVATE-restored-rank{rank}.json').write_text(json.dumps(dict(digest=restored)))
    del saved
    dist.barrier()
    if rank==0:
        states=[json.loads((a.output/f'PRIVATE-restored-rank{r}.json').read_text())['digest'] for r in range(world)]
        if set(states)!={expected}:raise AssertionError('Every restored replica must match original state')
        os.link(parent,a.output/'PRIVATE-u103.pt')
    dist.barrier()
    class Objective(nn.Module):
        def __init__(self,m,k):super().__init__();self.m=m;self.k=k
        def forward(self,x,y,mask):return target_loss_sum(self.m,x,y,mask,self.k,mode=p['workspace_mode'])
    compiled=torch.compile(Objective(model,2),backend='neuron',fullgraph=True,dynamic=False)
    data=ProbabilitySources(a.root/p['data_relative']);begin=(103*world+rank)*p['batch']
    x,y,mask=data.batch(list(range(begin,begin+p['batch'])))
    count=data.target_count(103*world*p['batch'],world*p['batch'])
    optimizer.zero_grad(set_to_none=True);tick=time.time()
    value=compiled(x.to('neuron'),y.to('neuron'),mask.to('neuron'))*(world/count);value.backward()
    for parameter in model.parameters():
        if parameter.grad is None:raise AssertionError('Missing registered gradient')
        dist.all_reduce(parameter.grad);parameter.grad.div_(world)
    score=value.detach().clone();dist.all_reduce(score);score/=world
    if rank==0:
        torch.save(dict(gradients={n:v.grad.detach().cpu() for n,v in model.named_parameters()},
            loss=float(score.cpu()),update=104,k=2),a.output/'PRIVATE-mature-gradient.pt')
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    update_seconds=time.time()-tick
    def save(state,state_hash):
        torch.save(dict(**state,digest=state_hash,update=104),a.output/'PRIVATE-u104.pt')
    capture_replicas(model,optimizer,rank=rank,world=world,rank_batch=p['checkpoint_rank_batch'],
        output=a.output,update=104,fallback_ops=list(torch_neuronx.get_fallback_ops()),on_primary=save)
    if rank==0:
        result=dict(status='COMPLETE_CHUNKED_CHECKPOINT_UPDATE_PENDING_AUDITS',world=world,
            actual_qualification_updates=1,scientific_optimizer_updates=0,retained_for_science=False,
            parameters=sum(v.numel() for v in model.parameters()),restored_all_replica_states_equal=True,
            all_replica_full_states_equal=True,source_targets=count,loss=float(score.cpu()),
            update_including_compilation_seconds=update_seconds,wall_seconds=time.time()-start,
            scope='One generated-slot update104 after exact full restoration of the original disposable update103 into every rank. Missing mature-gradient and independent restart checks remain separate; no scientific state reuse.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--corpus');main(p.parse_args())
