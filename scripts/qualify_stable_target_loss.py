"""Numerical arrays only:Native large-vocabulary CE gradients at high confidence."""
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
    from compute_trainium.training import file_sha256
    from compute_trainium.stable_target_loss import stable_target_log_probs
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='stable_target_loss_numerical_fixture' or plan['world']!=64 or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Exact fixture plan')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen fixture source changed')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world)
    torch.manual_seed(7543+rank);margins=[-30.,0.,8.,15.,20.,25.,30.,40.]
    cpu=torch.randn(len(margins),100352,dtype=torch.float32)*.1;targets=torch.tensor([3+rank+i*125 for i in range(len(margins))])
    cpu[torch.arange(len(margins)),targets]=torch.tensor(margins)
    probability=torch.softmax(cpu.double(),-1);expected=probability.clone();expected[torch.arange(len(margins)),targets]=0.
    expected[torch.arange(len(margins)),targets]=-expected.sum(-1)
    records={}
    for label in ['standard','stable']:
        logits=cpu.to('neuron').requires_grad_(True);y=targets.to('neuron')
        lp=torch.log_softmax(logits,-1).gather(-1,y[:,None]).squeeze(-1) if label=='standard' else stable_target_log_probs(logits,y)
        (-lp.sum()).backward();actual=logits.grad.cpu().double()
        error=((actual-expected).square().sum(-1)/expected.square().sum(-1).clamp_min(1e-300)).sqrt()
        records[label]=dict(row_relative_l2=error.tolist(),maximum_relative_l2=float(error.max()),losses=(-lp).detach().cpu().tolist())
        del logits,actual;torch.neuron.empty_cache()
    if records['stable']['maximum_relative_l2']>.02:raise AssertionError('Stable Native gradient exceeds unchanged2% oracle limit')
    fallback=torch_neuronx.get_fallback_ops()
    if fallback:raise AssertionError('Loss primitive used CPU fallback:'+str(fallback))
    dist.barrier();a.output.mkdir(parents=True,exist_ok=True)
    (a.output/f'rank{rank}-rows.json').write_text(json.dumps(records,indent=2)+'\n')
    if rank==0:(a.output/'report.json').write_text(json.dumps(dict(status='PASS_STABLE_TARGET_LOSS_NATIVE_FIXTURE',world=world,
        wall_seconds=time.time()-started,records=records,margins=margins,optimizer_updates=0,research_examples=0,fallback_ops=fallback),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
