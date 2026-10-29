"""Bounded stage timings from a qualified disposable checkpoint; no quality fit."""
import argparse
import json
import os
import statistics
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))]


def main(a):
    from pathlib import Path
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.interleaved_objective import make_model, target_loss_sum
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    p = json.loads(a.plan.read_text())
    src = Path(__file__).resolve().parents[1]
    if file_sha256(a.plan) != a.plan_sha or any(file_sha256(src/f) != h for f,h in p['worker_source_sha256'].items()):
        raise AssertionError('Registered profiling sources changed')
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    if world != 64 or p['purpose'] != 'interleaved_stage_profile':
        raise ValueError('Registered all-chip profiling only')
    torch.set_num_threads(1)
    os.environ['NEURON_RT_ROOT_COMM_ID'] = '127.0.0.1:'+str(a.comm_port)
    torch.neuron.set_device(0); dist.init_process_group('neuron')
    a.output.mkdir(parents=True, exist_ok=True)
    parent = a.root/p['checkpoint_relative']
    if any(file_sha256(parent/name)!=expected for name,expected in p['checkpoint_sha256'].items()):
        raise AssertionError('Qualified disposable checkpoint bytes changed')
    saved = torch.load(parent/'PRIVATE-u103.pt', weights_only=False, map_location='cpu', mmap=True)
    model = make_model(p, explicit_attention=True).to('neuron')
    optimizer = torch.optim.AdamW(parameter_groups(model, model.embedding.weight), lr=p['lr'],
                                 betas=(.9,.95), eps=1e-8, foreach=False, fused=True)
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['rng'])
    if digest(cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state()))) != saved['digest']:
        raise AssertionError('Profiling checkpoint restoration differs')
    del saved
    data = ProbabilitySources(a.root/p['data_relative'])
    class Objective(nn.Module):
        def __init__(self): super().__init__(); self.model=model
        def forward(self,x,y,mask):
            return target_loss_sum(self.model,x,y,mask,2,mode='latent')
    compiled = torch.compile(Objective(),backend='neuron',fullgraph=True,dynamic=False)
    records=[]; start=time.time()
    for update in range(104,115):
        row=dict(update=update); tick=time.time()
        base=(update-1)*world*p['batch']
        x,y,mask=data.batch(list(range(base+rank*p['batch'],base+(rank+1)*p['batch'])))
        count=data.target_count(base,world*p['batch'])
        optimizer.zero_grad(set_to_none=True)
        native=(x.to('neuron'),y.to('neuron'),mask.to('neuron')); torch.neuron.synchronize()
        row['data_and_transfer']=time.time()-tick; tick=time.time()
        value=compiled(*native)*(world/count); torch.neuron.synchronize()
        row['forward']=time.time()-tick; tick=time.time()
        value.backward(); torch.neuron.synchronize()
        row['backward']=time.time()-tick; tick=time.time()
        for parameter in model.parameters():
            if parameter.grad is None: raise AssertionError('Missing gradient')
            dist.all_reduce(parameter.grad); parameter.grad.div_(world)
        total=value.detach().clone(); dist.all_reduce(total); total/=world; torch.neuron.synchronize()
        row['collectives']=time.time()-tick; tick=time.time()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); torch.neuron.synchronize()
        row['optimizer']=time.time()-tick
        row['loss']=float(total.cpu()); row['gradient_norm']=float(norm.cpu())
        records.append(row)
        if update==104:
            actual=digest(cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state())))
            expected=torch.load(parent/'PRIVATE-u104.pt',weights_only=False,map_location='cpu',mmap=True)
            if actual!=expected['digest']: raise AssertionError('Timed code must exactly replay qualified update104')
            del expected
        if rank==0:(a.output/'progress.json').write_text(json.dumps(dict(update=update,seconds=time.time()-start)))
        dist.barrier()
    measured=records[1:]
    stages=('data_and_transfer','forward','backward','collectives','optimizer')
    summary={key:statistics.mean(row[key] for row in measured) for key in stages}
    values=torch.tensor([summary[key] for key in stages],device='neuron')
    dist.all_reduce(values,op=dist.ReduceOp.MAX)
    if rank==0:
        if set(torch_neuronx.get_fallback_ops())-{'aten::_to_copy'}:
            raise AssertionError('Unexpected device fallback')
        report=dict(status='COMPLETE_INTERLEAVED_STAGE_PROFILE',world=world,
                    qualified_update104_exact=True,actual_disposable_updates=11,
                    scientific_optimizer_updates=0,retained_for_science=False,
                    timed_updates=10,rank0_mean_seconds=summary,
                    maximum_rank_mean_seconds=dict(zip(stages,values.cpu().tolist())),
                    rank0_records=records,fallback_ops=list(torch_neuronx.get_fallback_ops()),
                    wall_seconds=time.time()-start,
                    scope='Explicit synchronization separates stages; these timings do not replace end-to-end throughput. No heldout data or scientific weights.')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser()
    for name in ('root','plan','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha',required=True);parser.add_argument('--comm-port',type=int,required=True)
    parser.add_argument('--corpus');main(parser.parse_args())
