"""Finite actual-shape Muon software benchmark;no model or research examples."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import torch
import torch.distributed as dist
from compute_trainium.owner_broadcast_muon_candidate import OwnerBroadcastMuon
from compute_trainium.parcae_runtime_scalars import RuntimeScalarMuon,changing_qualification_schedule
from compute_trainium.parcae_optimizer_precise import muon_tensor_step_fp32
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--variant',choices=['replicated','owner'],required=True)
    parser.add_argument('--comm-port',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus');args=parser.parse_args();started=time.time()
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Benchmark plan changed')
    plan=json.loads(args.plan.read_text())
    if (not plan['qualification_only'] or plan['optimizer_updates']!=120
            or plan['research_examples']!=0 or plan['root_comm_ports'][args.output.name]!=args.comm_port):
        raise ValueError('Exact bounded software-fixture benchmark required')
    source=Path(__file__).resolve().parents[1]
    if any(file_sha256(source/p)!=h for p,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen sources changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(args.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank=dist.get_rank()
    if dist.get_world_size()!=32:raise ValueError('Exact eight-chip32-rank geometry required')
    args.output.mkdir(exist_ok=True)
    generator=torch.Generator().manual_seed(41701);groups=[]
    for item in plan['matrix_groups']:
        parameters=[]
        for _ in range(item['count']):
            p=torch.nn.Parameter((torch.randn(item['shape'],generator=generator)*.02).to('neuron'))
            p.grad=(torch.randn(item['shape'],generator=generator)*.001).to('neuron');parameters.append(p)
        groups.append(dict(params=parameters,kind='muon',lr=1e-4,momentum=.95,weight_decay=.2,ns_steps=5))
    optimizer=(OwnerBroadcastMuon if args.variant=='owner' else RuntimeScalarMuon)(groups)
    optimizer.tensor_step=muon_tensor_step_fp32
    def state():
        return dict(parameters=[[p.detach() for p in g['params']] for g in optimizer.param_groups],optimizer=optimizer.state_dict())
    torch.neuron.synchronize();dist.barrier();warm_start=None
    for step in range(120):
        if step==20:torch.neuron.synchronize();dist.barrier();warm_start=time.time()
        changing_qualification_schedule(optimizer,step);optimizer.step()
        if step==119:torch.neuron.synchronize();warm_end=time.time()
        if step in [0,119]:
            torch.neuron.synchronize();fingerprint=digest(state())
            (args.output/f'state-u{step+1}-rank{rank}.json').write_text(json.dumps(dict(digest=fingerprint,fallback_ops=list(torch_neuronx.get_fallback_ops()),warm_seconds=(warm_end-warm_start) if step==119 else None))+'\n')
            dist.barrier()
            if rank==0:
                rows=[json.loads((args.output/f'state-u{step+1}-rank{i}.json').read_text()) for i in range(32)]
                if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in rows):raise AssertionError('Unexpected native fallback')
                if len({r['digest'] for r in rows})!=1:raise AssertionError('Complete optimizer/weight replicas differ')
                atomic_checkpoint(args.output/f'state-u{step+1}.pt',state())
            dist.barrier()
    torch.neuron.synchronize()
    if rank==0:
        report=dict(status='COMPLETE_OWNER_MUON_FIXTURE_BENCHMARK',variant=args.variant,
            plan_sha256=args.plan_sha,qualification_only=True,research_examples=0,research_optimizer_updates=0,
            optimizer_updates=120,timing_updates=100,warm_seconds=max(r['warm_seconds'] for r in rows),
            matrix_groups=plan['matrix_groups'],all_complete_replica_states_equal=True,
            first_state_sha256=file_sha256(args.output/'state-u1.pt'),
            final_state_sha256=file_sha256(args.output/'state-u120.pt'),
            fallback_ops=list(torch_neuronx.get_fallback_ops()),wall_seconds=time.time()-started,
            scope='Actual-shape matrix-only software fixtures with fixed identical gradients and changing schedules.100 warm timed Muon updates exclude model forwards,AdamW,gradient reduction,checkpoint copies and serving. Native cross-variant state comparison and full-training performance remain separate gates. No scientific adoption.')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
