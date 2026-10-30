"""ABBA whole-update qualification, isolated from every scientific trainer."""
import argparse
import gc
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
    from compute_trainium.bucketed_gradient_reduction import reduce_and_clip
    from compute_trainium.bounded_replica_capture import capture_replicas
    from compute_trainium.interleaved_objective import target_loss_sum
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.tied_interleaved_ordinary import make_model
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    p=json.loads(a.plan.read_text());src=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(src/f)!=h for f,h in p['worker_source_sha256'].items()):
        raise AssertionError('Frozen source/plan changed')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=64 or p['purpose']!='bucketed_gradient_disposable_qualification':raise ValueError('Registered fixture only')
    torch.set_num_threads(1);os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    torch.neuron.set_device(0);dist.init_process_group('neuron')
    a.output.mkdir(parents=True,exist_ok=True);started=time.time()
    data=ProbabilitySources(a.root/p['data_relative'])
    if data.context!=512:raise AssertionError('Actual training context required')
    torch.manual_seed(p['seed']);model=make_model(p,explicit_attention=True)
    initial={n:v.clone() for n,v in model.state_dict().items()}
    if a.mode=='warm' and rank==0:torch.save(initial,a.output/'PRIVATE-initial.pt')
    model.to('neuron')
    class Objective(torch.nn.Module):
        def __init__(self,m):super().__init__();self.model=m
        def forward(self,x,y,mask):return target_loss_sum(self.model,x,y,mask,0,mode='ordinary')
    compiled=torch.compile(Objective(model),backend='neuron',fullgraph=True,dynamic=False)
    records=[];opt=None
    trials=p['trial_order'] if a.mode=='warm' else ['replay']
    for trial in trials:
        if opt is not None:opt.zero_grad(set_to_none=True);del opt;gc.collect()
        model.load_state_dict(initial)
        opt=torch.optim.AdamW(parameter_groups(model,model.embedding.weight,decay=.1),
            lr=p['lr'],betas=(.9,.95),eps=1e-8,foreach=False,fused=True)
        torch.manual_seed(p['seed']+1)
        if a.mode=='replay':
            saved=torch.load(a.parent/'PRIVATE-bucketed1-u103.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
            if digest(cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())))!=saved['digest']:
                raise AssertionError('Complete model/optimizer/RNG restoration differs')
            del saved
        dist.barrier();trial_started=time.time();telemetry=[];warm_started=None;warm_seconds=None
        bucketed=trial.startswith('bucketed') or a.mode=='replay'
        updates=range(104,105) if a.mode=='replay' else range(1,105 if trial=='bucketed1' else 104)
        for update in updates:
            tick=time.time();base=(update-1)*world*p['batch']
            if update==4:warm_started=tick
            x,y,mask=data.batch(list(range(base+rank*p['batch'],base+(rank+1)*p['batch'])))
            count=data.target_count(base,world*p['batch'])
            opt.zero_grad(set_to_none=True)
            value=compiled(x.to('neuron'),y.to('neuron'),mask.to('neuron'))*(world/count);value.backward()
            capture=(a.mode=='warm' and ((trial in ('original1','bucketed1') and update==1) or (trial=='bucketed1' and update==104)))
            # Capture the globally averaged gradient before clipping. Reconstruct
            # the norm coefficient only for captures, outside timed warm updates.
            if bucketed:
                averaged={}
                if capture:
                    # Separately save local gradients to reproduce the same
                    # averaging with an unclipped temporary parameter list.
                    # Avoid this extra work in all timed warm updates.
                    shadow=[torch.nn.Parameter(torch.empty_like(v),requires_grad=True) for v in model.parameters()]
                    for q,v in zip(shadow,model.parameters()):q.grad=v.grad.detach().clone()
                    reduce_and_clip(shadow,dist.all_reduce,world,max_norm=1e30,bucket_bytes=p['bucket_bytes'])
                    if rank==0:averaged={n:q.grad.detach().cpu() for (n,_),q in zip(model.named_parameters(),shadow)}
                    del shadow
                norm,buckets=reduce_and_clip(model.parameters(),dist.all_reduce,world,bucket_bytes=p['bucket_bytes'])
            else:
                for v in model.parameters():
                    if v.grad is None:raise AssertionError('Missing registered parameter gradient')
                    dist.all_reduce(v.grad);v.grad.div_(world)
                averaged={n:v.grad.detach().cpu() for n,v in model.named_parameters()} if capture and rank==0 else {}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);buckets=len(list(model.parameters()))
            total=value.detach().clone();dist.all_reduce(total);total/=world
            if capture and rank==0:
                torch.save(dict(gradients=averaged,loss=float(total.cpu()),update=update),a.output/f'PRIVATE-{trial}-gradient-u{update}.pt')
            opt.step();loss=float(total.cpu());norm_value=float(norm.cpu())
            dist.barrier()
            seconds=time.time()-tick
            if update==103:warm_seconds=time.time()-warm_started
            if rank==0:
                telemetry.append(dict(update=update,seconds=seconds,loss=loss,gradient_norm=norm_value,buckets=buckets,source_targets=count))
                (a.output/'progress.json').write_text(json.dumps(dict(trial=trial,update=update,wall_seconds=time.time()-started)))
                if update==12 and sum(r['seconds'] for r in telemetry[-9:])/9>p['early_warm_mean_seconds_limit']:
                    raise RuntimeError('Whole-update early throughput bound')
            if (a.mode=='warm' and update==1 and trial in ('original1','bucketed1')):
                dist.barrier()
                if rank==0:
                    state=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
                    torch.save(dict(**state,digest=digest(state),update=update),a.output/f'PRIVATE-{trial}-u1.pt');del state
                dist.barrier()
            if (trial=='bucketed1' and update in (103,104)) or a.mode=='replay':
                def save_or_compare(state,state_hash):
                    if a.mode=='warm':torch.save(dict(**state,digest=state_hash,update=update),a.output/f'PRIVATE-bucketed1-u{update}.pt')
                    else:
                        expected=torch.load(a.parent/'PRIVATE-bucketed1-u104.pt',map_location='cpu',mmap=True,weights_only=False)
                        if expected['digest']!=state_hash:raise AssertionError('Fresh-process complete-state replay differs')
                capture_replicas(model,opt,rank=rank,world=world,rank_batch=4,output=a.output,
                    update=update,fallback_ops=list(torch_neuronx.get_fallback_ops()),on_primary=save_or_compare)
            dist.barrier()
        if rank==0:
            warm=[r for r in telemetry if 4<=r['update']<=103]
            records.append(dict(trial=trial,updates=telemetry,warm_updates=len(warm),
                mean_warm_seconds=sum(r['seconds'] for r in warm)/len(warm) if warm else None,
                warm_block_seconds=warm_seconds,
                wall_seconds=time.time()-trial_started))
            (a.output/'trials.json').write_text(json.dumps(records))
        fallback=list(torch_neuronx.get_fallback_ops())
        if set(fallback)-{'aten::_to_copy'}:raise AssertionError('Unexpected fallback')
        dist.barrier()
    if rank==0:
        result=dict(status='COMPLETE_BUCKETED_ABBA_PENDING_AUDIT' if a.mode=='warm' else 'PASS_BUCKETED_EXACT_REPLAY',
            world=world,trials=records,wall_seconds=time.time()-started,
            disposable_optimizer_updates=sum(len(r['updates']) for r in records),scientific_optimizer_updates=0,
            retained_for_science=False,fallback_ops=fallback)
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


def parser():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    # The common isolated launcher supplies this compatibility argument. The
    # registered ProbabilitySources path remains the sole qualification data.
    p.add_argument('--corpus',type=Path)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--mode',choices=['warm','replay'],required=True);p.add_argument('--parent',type=Path)
    return p


if __name__=='__main__':main(parser().parse_args())
