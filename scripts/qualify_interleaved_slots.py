"""Actual-shape, disposable natural-TRAIN throughput and state qualification."""
import argparse,json,os,time
from pathlib import Path

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main(a):
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_slots_objective import make_model as probability_model
    from compute_trainium.interleaved_workspace import matrix_work
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.stable_target_loss import stable_target_log_probs
    from compute_trainium.interleaved_objective import target_loss_sum,reference_loss_sum
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    from scripts.qualify_trained_gradient import compare_gradients
    from compute_trainium.probability_model_factory import gradient_families
    p=json.loads(a.plan.read_text());src=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(src/f)!=h for f,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen warm sources changed')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
    if world!=64 or p['purpose']!='interleaved_workspace_actual_shape_qualification':raise ValueError('Registered actual-shape fixture only')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port);torch.neuron.set_device(0);dist.init_process_group('neuron')
    a.output.mkdir(parents=True,exist_ok=True);started=time.time();data=ProbabilitySources(a.root/p['data_relative'])
    c=ModelConfig(**p['model_config']);torch.manual_seed(p['seed']);model=probability_model(p,explicit_attention=True)
    initial={n:v.clone() for n,v in model.state_dict().items()};model.to('neuron')
    opt=torch.optim.AdamW(parameter_groups(model,model.embedding.weight),lr=p['lr'],betas=(.9,.95),eps=1e-8,foreach=False,fused=True)
    torch.manual_seed(p['seed']+1)
    class Objective(nn.Module):
        def __init__(self,m,k):super().__init__();self.m=m;self.k=k
        def forward(self,x,y,mask):return target_loss_sum(self.m,x,y,mask,self.k,mode=p['workspace_mode'])
    compiled={k:torch.compile(Objective(model,k),backend='neuron',fullgraph=True,dynamic=False) for k in p.get('cold_local_supported_steps',[0,1,2])}
    def inputs(update,worker=rank):
        base=(update-1)*world*p['batch'];indices=list(range(base+worker*p['batch'],base+(worker+1)*p['batch']))
        return data.batch(indices)
    def numerical(gradients,ref):
        result={'overall':compare_gradients(gradients,ref)}
        for family,match in gradient_families(p):
            result[family]=compare_gradients({n:v for n,v in gradients.items() if match(n)},{n:v for n,v in ref.items() if match(n)})
        return result
    checks={};telemetry=[];start_update=1
    if a.mode=='warm':
        # Local actual-shape CPU comparison before any actual-shape update.
        x,y,mask=inputs(1);native=(x.to('neuron'),y.to('neuron'),mask.to('neuron'))
        for k in p.get('cold_local_supported_steps',[0,1,2]):
            opt.zero_grad(set_to_none=True);value=compiled[k](*native)/mask.sum().item();value.backward()
            if rank==0:
                gradients={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()}
                torch.set_num_threads(8);ref=probability_model(p);ref.load_state_dict(initial)
                loss=reference_loss_sum(ref,x,y,mask,k,mode=p['workspace_mode'])/mask.sum()
                loss.backward();check=numerical(gradients,{n:v.grad for n,v in ref.named_parameters()})
                check['loss_error']=float(value.detach().cpu())-float(loss.detach());checks[str(k)]=check
                (a.output/'cold-local-checks.json').write_text(json.dumps(checks,indent=2))
                if max(check[f]['relative_l2'] for f in ('overall',*[n for n,_ in gradient_families(p)]))>.02 or abs(check['loss_error'])>.003:
                    raise AssertionError('Actual-shape CPU local gate failed')
                del ref,loss,gradients;torch.set_num_threads(1)
            dist.barrier()
        opt.zero_grad(set_to_none=True);torch.manual_seed(p['seed']+1)
        del native,x,y,mask
    else:
        saved=torch.load(a.parent/'PRIVATE-u103.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        if digest(cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())))!=saved['digest']:
            raise AssertionError('Full-state restoration differs')
        start_update=104;del saved
    del initial
    for update in range(start_update,105):
        profiler=None
        if update==2 and rank==0:
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
            profiler.__enter__()
        tick=time.time();schedule=p.get('depth_schedule',[0,1,2]);k=schedule[(update-1)%len(schedule)]
        if update==104:k=p.get('replay_depth',2)
        x,y,mask=inputs(update);count=data.target_count((update-1)*world*p['batch'],world*p['batch'])
        opt.zero_grad(set_to_none=True);value=compiled[k](x.to('neuron'),y.to('neuron'),mask.to('neuron'))*(world/count);value.backward()
        for parameter in model.parameters():
            if parameter.grad is None:raise AssertionError('Every registered parameter requires a gradient')
            dist.all_reduce(parameter.grad);parameter.grad.div_(world)
        total=value.detach().clone();dist.all_reduce(total);total/=world
        if update==104 and rank==0:
            torch.save(dict(gradients={n:v.grad.detach().cpu() for n,v in model.named_parameters()},loss=float(total.cpu()),update=update,k=k),a.output/'PRIVATE-mature-gradient.pt')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        loss=float(total.cpu());seconds=time.time()-tick
        if profiler is not None:
            profiler.__exit__(None,None,None)
            profiler.export_chrome_trace(str(a.output/'PRIVATE-cpu-profile.json'))
            (a.output/'cpu-profile-summary.txt').write_text(profiler.key_averages().table(sort_by='cpu_time_total',row_limit=40))
        if rank==0:
            telemetry.append(dict(update=update,extra_steps=k,loss=loss,gradient_norm=float(norm.cpu()),seconds=seconds,source_targets=count,
                nominal_training_matrix_flops=world*matrix_work(c,data.context,k,mode=p['workspace_mode'],batch=p['batch'])['training_matmul_flops']))
            (a.output/'updates.json').write_text(json.dumps(telemetry))
            (a.output/'progress.json').write_text(json.dumps(dict(update=update,seconds=time.time()-started)))
            if update==12 and sum(r['seconds'] for r in telemetry[-9:])/9>p['early_warm_mean_seconds_limit']:
                raise RuntimeError('Registered early throughput screen failed')
        if update in (103,104) and p.get('checkpoint_rank_batch'):
            from compute_trainium.bounded_replica_capture import capture_replicas
            def save_or_compare(state,state_hash):
                if a.mode=='warm':torch.save(dict(**state,digest=state_hash,update=update),a.output/f'PRIVATE-u{update}.pt')
                else:
                    expected=torch.load(a.parent/'PRIVATE-u104.pt',weights_only=False,map_location='cpu',mmap=True)
                    if state_hash!=expected['digest']:raise AssertionError('Grouped capture/replay differs from original complete state')
            capture_replicas(model,opt,rank=rank,world=world,rank_batch=p['checkpoint_rank_batch'],output=a.output,
                update=update,fallback_ops=list(torch_neuronx.get_fallback_ops()),on_primary=save_or_compare)
        elif update in (103,104):
            state=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()));state_hash=digest(state)
            (a.output/f'replica-{rank}.json').write_text(json.dumps(dict(update=update,digest=state_hash,fallback_ops=list(torch_neuronx.get_fallback_ops()))))
            dist.barrier()
            if rank==0:
                rows=[json.loads((a.output/f'replica-{r}.json').read_text()) for r in range(world)]
                if len({r['digest'] for r in rows})!=1:raise AssertionError('Full replica states differ')
                if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in rows):raise AssertionError('Unexpected fallback')
                if a.mode=='warm':torch.save(dict(**state,digest=state_hash,update=update),a.output/f'PRIVATE-u{update}.pt')
                else:
                    expected=torch.load(a.parent/'PRIVATE-u104.pt',weights_only=False,map_location='cpu')
                    if state_hash!=expected['digest']:raise AssertionError('Separate-process model/optimizer/RNG replay differs')
            del state
        dist.barrier()
    if rank==0:
        warm=[r for r in telemetry if 4<=r['update']<=103]
        result=dict(status='COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if a.mode=='warm' else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY',
            world=world,mode=a.mode,actual_qualification_updates=len(telemetry),scientific_optimizer_updates=0,
            retained_for_science=False,parameters=sum(v.numel() for v in model.parameters()),warm_updates=len(warm),
            mean_warm_seconds=None if not warm else sum(r['seconds'] for r in warm)/len(warm),
            warm_by_depth={str(k):sum(r['seconds'] for r in warm if r['extra_steps']==k)/max(1,sum(r['extra_steps']==k for r in warm)) for k in p.get('cold_local_supported_steps',[0,1,2])},
            wall_seconds=time.time()-started,all_replica_full_states_equal=True,cold_local_checks=checks)
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--mode',choices=['warm','replay'],required=True);p.add_argument('--parent',type=Path);main(p.parse_args())
