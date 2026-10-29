"""All-chip disposable numerical fixture for persistent latent tokens."""
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
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_workspace import InterleavedWorkspace
    from compute_trainium.probability_workspace import ProbabilityWorkspace
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.stable_target_loss import stable_target_log_probs
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from scripts.qualify_trained_gradient import compare_gradients
    from compute_trainium.probability_model_factory import gradient_families
    p=json.loads(a.plan.read_text());src=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(src/f)!=h for f,h in p['worker_source_sha256'].items()):
        raise AssertionError('Frozen fixture sources differ')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
    if world!=64 or p['purpose']!='interleaved_workspace_software_qualification':raise ValueError('Exact fixture required')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    torch.neuron.set_device(0);dist.init_process_group('neuron');a.output.mkdir(parents=True,exist_ok=True)
    start=time.time();config=ModelConfig(**p['model_config']);torch.manual_seed(p['seed'])
    model=InterleavedWorkspace(config,explicit_attention=True)
    initial={n:v.clone() for n,v in model.state_dict().items()}
    reference=None
    if rank==0:
        reference=InterleavedWorkspace(config);reference.load_state_dict(initial)
    # Constructing the independent reference must not alter replica RNG state.
    torch.manual_seed(p['seed']+1)
    def optimizer(m,fused):
        return torch.optim.AdamW(parameter_groups(m,m.embedding.weight),lr=3e-4,betas=(.9,.95),eps=1e-8,foreach=False,fused=fused)
    model.to('neuron');opt=optimizer(model,True)
    ropt=optimizer(reference,False) if rank==0 else None
    if a.mode=='replay':
        saved=torch.load(a.parent/'PRIVATE-u4.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        if digest(cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())))!=saved['digest']:
            raise AssertionError('Exact full-state reload mismatch')
        if rank==0:
            r=torch.load(a.parent/'PRIVATE-reference-u4.pt',weights_only=False);reference.load_state_dict(r['model']);ropt.load_state_dict(r['optimizer'])
        updates=[5]
    else:updates=list(range(1,6))
    class Objective(nn.Module):
        def __init__(self,m,steps,proposal):super().__init__();self.model=m;self.steps=steps;self.proposal=proposal
        def forward(self,x,y):return -stable_target_log_probs(self.model(x,self.steps,self.proposal),y).mean()
    compiled={};records=[]
    for update in updates:
        steps,proposal=p['update_cases'][update-1];key=(steps,proposal)
        if key not in compiled:compiled[key]=torch.compile(Objective(model,*key),backend='neuron',fullgraph=True,dynamic=False)
        g=torch.Generator().manual_seed(p['seed']+100+update)
        allx=torch.randint(config.vocab_size,(world,p['length']),generator=g)
        ally=torch.randint(config.vocab_size,(world,p['length']),generator=g)
        before={n:v.detach().cpu().clone() for n,v in model.named_parameters()} if rank==0 else None
        opt.zero_grad(set_to_none=True)
        value=compiled[key](allx[rank:rank+1].contiguous().to('neuron'),ally[rank:rank+1].contiguous().to('neuron'))
        value.backward()
        for parameter in model.parameters():
            if parameter.grad is None:raise AssertionError('Missing gradient')
            dist.all_reduce(parameter.grad);parameter.grad.div_(world)
        scalar=value.detach().clone();dist.all_reduce(scalar);scalar/=world
        gradients={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()} if rank==0 else None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        state=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
        state_digest=digest(state)
        (a.output/f'replica-{rank}.json').write_text(json.dumps(dict(digest=state_digest,fallback_ops=list(torch_neuronx.get_fallback_ops()))))
        dist.barrier()
        if rank==0:
            replicas=[json.loads((a.output/f'replica-{i}.json').read_text()) for i in range(world)]
            if len({r['digest'] for r in replicas})!=1:raise AssertionError('Replica full states differ')
            if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):raise AssertionError('Unexpected Native fallback')
            ropt.zero_grad(set_to_none=True);r_before={n:v.detach().clone() for n,v in reference.named_parameters()}
            # Independent SDPA and native PyTorch cross entropy.
            rlogits=reference(allx,steps,proposal);rloss=torch.nn.functional.cross_entropy(rlogits.flatten(0,1),ally.flatten());rloss.backward()
            rgrad={n:v.grad.clone() for n,v in reference.named_parameters()}
            comparison=compare_gradients(gradients,rgrad)
            families={family:compare_gradients({n:v for n,v in gradients.items() if match(n)},
                        {n:v for n,v in rgrad.items() if match(n)}) for family,match in
                        gradient_families(p)}
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True);ropt.step()
            native_delta={n:state['model'][n]-before[n] for n in before}
            reference_delta={n:v.detach()-r_before[n] for n,v in reference.named_parameters()}
            step_compare=compare_gradients(native_delta,reference_delta)
            cmom={str(i)+':'+k:v for i,s in ropt.state_dict()['state'].items() for k,v in s.items() if k!='step'}
            nmom={str(i)+':'+k:v for i,s in state['optimizer']['state'].items() for k,v in s.items() if k!='step'}
            moments=compare_gradients(nmom,cmom)
            row=dict(update=update,extra_steps=steps,proposal=proposal,gradient=comparison,families=families,
                optimizer_update=step_compare,moments=moments,loss_error=float(scalar.cpu())-float(rloss.detach()),
                global_gradient_norm=float(norm.cpu()),all_replica_full_states_equal=True)
            records.append(row);(a.output/'progress.json').write_text(json.dumps(records,indent=2))
            if max([comparison['relative_l2'],step_compare['relative_l2'],moments['relative_l2'],*[v['relative_l2'] for v in families.values()]])>.02 or abs(row['loss_error'])>.003:
                raise AssertionError('Original 2% gradient/update/moment or .003 loss gate failed')
            if update==4:
                torch.save(dict(**state,digest=state_digest),a.output/'PRIVATE-u4.pt')
                torch.save(dict(model=reference.state_dict(),optimizer=ropt.state_dict()),a.output/'PRIVATE-reference-u4.pt')
            if update==5:
                if a.mode=='replay':
                    expected=torch.load(a.parent/'PRIVATE-u5.pt',weights_only=False)
                    row['separate_process_replay_exact']=digest(state)==expected['digest']
                    if not row['separate_process_replay_exact']:raise AssertionError('Full optimizer/model/RNG replay mismatch')
                else:torch.save(dict(**state,digest=state_digest),a.output/'PRIVATE-u5.pt')
        dist.barrier()
    if rank==0:
        report=dict(status='PASS_INTERLEAVED_WORKSPACE_SOFTWARE_FIXTURE',world=world,mode=a.mode,records=records,
            actual_fixture_updates=len(updates),research_examples=0,research_optimizer_updates=0,
            parameters=sum(v.numel() for v in model.parameters()),wall_seconds=time.time()-start,scientific_fit_admitted=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--root');p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--mode',choices=['warm','replay'],required=True);p.add_argument('--parent',type=Path);main(p.parse_args())
