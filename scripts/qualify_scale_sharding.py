"""Bounded shared-weight/tied-table sharding fixture; zero research examples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def worker(rank, world, backend, output, mode='warm', parent=None, comm_port=None):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.scale_sharded_candidate import shard_model, ShardedExactOptimizer, local_parameter
    from compute_trainium.scale_sharded_fixture import SoftwareModel, loss, cpu_step, gather_named, full_moments, cpu_moments, cpu_tree, relative
    from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
    torch.set_num_threads(1)
    started = time.time();output = Path(output);output.mkdir(parents=True, exist_ok=True)
    if backend == 'neuron':
        os.environ['NEURON_RT_ROOT_COMM_ID'] = '127.0.0.1:' + str(comm_port)
        import torch_neuronx
        torch.neuron.set_device(0)
        device = 'neuron'
    else: device = 'cpu'
    dist.init_process_group(backend, rank=rank, world_size=world)
    mesh = init_device_mesh(device, (world,))
    torch.manual_seed(92017)
    model = SoftwareModel().to(device)
    reference = None
    if rank == 0:
        torch.manual_seed(92017);reference = SoftwareModel()
        cpu_optimizer = ParcaeRuntimeScalarOptimizer(reference,gate_policy='release_muon',lr=1e-4,muon_lr=1e-4)
    model = shard_model(model, mesh)
    if model.transformer.wte._parameters['weight'] is not model.lm_head._parameters['weight']:
        raise AssertionError('Sharding broke tied input/output alias')
    optimizer = ShardedExactOptimizer(model,lr=1e-4,muon_lr=1e-4)
    if mode == 'replay':
        saved = torch.load(Path(parent)/f'rank{rank}-u2.pt',map_location='cpu',weights_only=False)
        if saved['rank']!=rank or saved['world']!=world or saved['purpose']!='scale_sharded_qualification_only':
            raise ValueError('Exact topology-bound qualification shard required')
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                local_parameter(parameter).copy_(saved['model_shards'][name].to(device))
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'])
        if rank == 0:
            cpu_saved = torch.load(Path(parent)/'cpu-u2.pt',map_location='cpu',weights_only=False)
            reference.load_state_dict(cpu_saved['model']);cpu_optimizer.load_state_dict(cpu_saved['optimizer'])
        updates = [3]
    else: updates = [1,2,3]
    records=[]
    for update in updates:
        optimizer.zero_grad()
        initial = gather_named(model)
        value = loss(model,rank,world,update,device)
        value.backward();norm = optimizer.finish_gradients()
        gradients = gather_named(model, gradients=True)
        scalar=value.detach().clone();dist.all_reduce(scalar)
        optimizer.step()
        actual = gather_named(model);moments = full_moments(optimizer)
        if rank == 0:
            before = {n:p.detach().clone() for n,p in reference.named_parameters()}
            cpu_loss,cpu_norm,cpu_gradient = cpu_step(reference,cpu_optimizer,world,update)
            wanted = {n:p.detach().clone() for n,p in reference.named_parameters()}
            actual_update={n:actual[n]-initial[n] for n in actual}
            expected_update={n:wanted[n]-before[n] for n in wanted}
            row=dict(update=update,gradient=relative(cpu_gradient,gradients),
                parameter_update=relative(expected_update,actual_update),weights=relative(wanted,actual),
                moments=relative(cpu_moments(cpu_optimizer,list(reference.named_parameters())),moments),
                loss_difference=float(scalar.cpu())-cpu_loss,global_norm_difference=norm-cpu_norm)
            if (max(row[k]['largest_named_relative_l2'] for k in ['gradient','parameter_update','moments']) > .02
                    or abs(row['loss_difference']) > .003):
                (output/'failed-comparison.json').write_text(json.dumps(row,indent=2)+'\n')
                raise AssertionError('Independent CPU sharded fixture numerical gate failed')
            if mode == 'replay':
                previous=torch.load(Path(parent)/'expected-u3.pt',map_location='cpu',weights_only=False)
                row['separate_process_replay_weights']=relative(previous['weights'],actual)
                row['separate_process_replay_moments']=relative(previous['moments'],moments)
                if max(row[k]['relative_l2'] for k in ['separate_process_replay_weights','separate_process_replay_moments'])>1e-6:
                    raise AssertionError('Separate-process sharded replay failed')
            records.append(row)
            (output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        if update == 2:
            payload=dict(purpose='scale_sharded_qualification_only',eligible_for_science=False,
                rank=rank,world=world,update=2,model_shards={n:local_parameter(p) for n,p in model.named_parameters()},
                optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state())
            torch.save(cpu_tree(payload),output/f'rank{rank}-u2.pt')
            if rank==0:
                torch.save(dict(model=reference.state_dict(),optimizer=cpu_optimizer.state_dict()),output/'cpu-u2.pt')
        if update == 3 and rank == 0:
            torch.save(dict(weights=actual,moments=moments),output/'expected-u3.pt')
        dist.barrier()
    if rank == 0:
        row=dict(status='PASS_SCALE_SHARDED_SOFTWARE_FIXTURE',backend=backend,world=world,mode=mode,
            tied_parameter_alias_preserved=True,shared_core_applications=2,records=records,
            actual_optimizer_updates_per_rank=len(updates),research_examples=0,research_optimizer_updates=0,
            scientific_fit_admitted=False,wall_seconds=time.time()-started,
            scope='Small numerical arrays only. Qualifies this fixture,not large-model training,precision or throughput.')
        (output/'report.json').write_text(json.dumps(row,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path);parser.add_argument('--corpus')
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--mode',choices=['warm','replay'],required=True);parser.add_argument('--parent',type=Path)
    parser.add_argument('--comm-port',type=int,required=True);args=parser.parse_args()
    from compute_trainium.training import file_sha256
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Frozen fixture plan changed')
    plan=json.loads(args.plan.read_text());source=Path(__file__).resolve().parents[1]
    if (plan['purpose']!='scale_sharded_software_fixture_only' or plan['world']!=64
            or plan['phases'][args.output.name]['mode']!=args.mode
            or plan['root_comm_ports'][args.output.name]!=args.comm_port):
        raise ValueError('Exact registered all-chip numerical fixture required')
    if any(file_sha256(source/f)!=sha for f,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Frozen source changed')
    worker(int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),'neuron',args.output,args.mode,args.parent,args.comm_port)


if __name__=='__main__':main()
