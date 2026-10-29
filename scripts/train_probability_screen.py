"""Finite, uniformly supervised probability-replay learning screen.

This worker does no held-out scoring. A controller must admit the frozen budget
after actual-shape qualification, and audit checkpoints before evaluating them.
"""
import argparse,json,math,os,time
from pathlib import Path

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

def learning_rate(update,arm,peak):
    if update<=arm['warmup_updates']:return peak*update/arm['warmup_updates']
    if update<=arm['decay_start']:return peak
    progress=(update-arm['decay_start'])/(len(arm['depth_schedule'])-arm['decay_start'])
    return peak*(.1+.9*.5*(1+math.cos(math.pi*progress)))


def main(a):
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_model_factory import probability_model
    from compute_trainium.probability_workspace import ProbabilityWorkspace,matrix_work
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.stable_target_loss import stable_target_log_probs
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(source/f)!=h for f,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen screen changed')
    if p['purpose']!='probability_workspace_learning_screen' or not p['scientific_fit_admitted']:raise ValueError('A new measured learning registration is required')
    arm=p['arms'][a.arm];rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=64:raise AssertionError('Registered global topology')
    torch.set_num_threads(1);os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    torch.neuron.set_device(0);dist.init_process_group('neuron');a.output.mkdir(parents=True,exist_ok=True)
    c=ModelConfig(**p['model_config']);data=ProbabilitySources(a.root/p['data_relative']);torch.manual_seed(p['seed'])
    model=probability_model(p,explicit_attention=True);initial_state=model.state_dict();initial_digest=digest(initial_state)
    shared_initial_digest=digest(type(initial_state)((n,v) for n,v in initial_state.items() if n!='remaining_budget_cues'))
    parameter_count=sum(v.numel() for v in model.parameters());del initial_state;model.to('neuron')
    opt=torch.optim.AdamW(parameter_groups(model,model.embedding.weight,decay=p['weight_decay']),lr=p['lr'],betas=(.9,.95),eps=1e-8,foreach=False,fused=True)
    torch.manual_seed(p['seed']+1);proposal=arm['proposal'];schedule=arm['depth_schedule'];telemetry=[];started=time.time();exposure=0;work=0
    class Objective(nn.Module):
        def __init__(self,m,k):super().__init__();self.m=m;self.k=k
        def forward(self,x,y,mask):return -(stable_target_log_probs(self.m(x,self.k,proposal),y)*mask).sum()
    compiled={k:torch.compile(Objective(model,k),backend='neuron',fullgraph=True,dynamic=False) for k in sorted(set(schedule))}
    def batch(update,worker=rank):
        begin=(update-1)*world*p['batch']+worker*p['batch']
        return data.batch(list(range(begin,begin+p['batch'])))
    checkpoints=[]
    for update,k in enumerate(schedule,1):
        lr=learning_rate(update,arm,p['lr'])
        for group in opt.param_groups:group['lr']=lr
        tick=time.time();x,y,mask=batch(update);count=data.target_count((update-1)*world*p['batch'],world*p['batch'])
        opt.zero_grad(set_to_none=True)
        value=compiled[k](x.to('neuron'),y.to('neuron'),mask.to('neuron'))*(world/count);value.backward()
        for parameter in model.parameters():
            if parameter.grad is None:raise AssertionError('Missing trainable gradient')
            dist.all_reduce(parameter.grad);parameter.grad.div_(world)
        total=value.detach().clone();dist.all_reduce(total);total/=world
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        loss=float(total.cpu());seconds=time.time()-tick;exposure+=count
        ops=world*matrix_work(c,data.context,k,proposal,batch=p['batch'])['training_matmul_flops'];work+=ops
        if rank==0:
            telemetry.append(dict(update=update,extra_steps=k,loss=loss,lr=lr,gradient_norm=float(norm.cpu()),seconds=seconds,real_targets=count,
                                  cumulative_real_targets=exposure,cumulative_matrix_flops=work))
            if update%16==0 or update in arm['checkpoint_updates']:
                (a.output/'updates.json').write_text(json.dumps(telemetry));(a.output/'progress.json').write_text(json.dumps(telemetry[-1]))
            if update==16 and sum(r['seconds'] for r in telemetry[-9:])/9>p['early_mean_seconds_limit']:
                raise RuntimeError('Registered learning throughput ceiling')
        if update in set(arm['checkpoint_updates'])|set(arm.get('recovery_checkpoint_updates',[])):
            if p.get('checkpoint_rank_batch'):
                from compute_trainium.bounded_replica_capture import capture_replicas
                pending=a.output/f'PRIVATE-u{update}-pending.pt'
                def save_primary(state,state_hash):
                    torch.save(dict(**state,digest=state_hash,update=update,real_targets=exposure,matrix_flops=work,
                        initial_digest=initial_digest,arm=a.arm,plan_sha256=a.plan_sha),pending)
                h=capture_replicas(model,opt,rank=rank,world=world,rank_batch=p['checkpoint_rank_batch'],output=a.output,
                    update=update,fallback_ops=list(torch_neuronx.get_fallback_ops()),on_primary=save_primary)
                if rank==0:
                    pending.rename(a.output/f'PRIVATE-u{update}.pt')
                    checkpoints.append(dict(update=update,real_targets=exposure,matrix_flops=work,full_state_digest=h))
                dist.barrier()
            else:
                # Checkpoint is the state before its independent, zero-update audit.
                state=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
                h=digest(state);(a.output/f'replica-{rank}.json').write_text(json.dumps(dict(update=update,digest=h,fallback_ops=list(torch_neuronx.get_fallback_ops()))))
                dist.barrier()
                if rank==0:
                    rows=[json.loads((a.output/f'replica-{i}.json').read_text()) for i in range(world)]
                    if len({r['digest'] for r in rows})!=1 or any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in rows):raise AssertionError('Replica/fallback gate')
                    checkpoint=dict(**state,digest=h,update=update,real_targets=exposure,matrix_flops=work,initial_digest=initial_digest,arm=a.arm,plan_sha256=a.plan_sha)
                    torch.save(checkpoint,a.output/f'PRIVATE-u{update}.pt')
                    checkpoints.append(dict(update=update,real_targets=exposure,matrix_flops=work,full_state_digest=h))
                del state
        if update in arm['checkpoint_updates']:
            # Capture the same fixed source batch for every endpoint and exit.
            # These TRAIN windows are reserved from the fitted prefix and do
            # not select a model or a duration.
            audit_start=p['audit_window_start']+rank*p['batch']
            ax,ay,am=data.batch(list(range(audit_start,audit_start+p['batch'])))
            audit_count=data.target_count(p['audit_window_start'],world*p['batch'])
            for depth in arm['audit_depths']:
                opt.zero_grad(set_to_none=True)
                audit=compiled[depth](ax.to('neuron'),ay.to('neuron'),am.to('neuron'))*(world/audit_count);audit.backward()
                for parameter in model.parameters():dist.all_reduce(parameter.grad);parameter.grad.div_(world)
                score=audit.detach().clone();dist.all_reduce(score);score/=world
                if rank==0:
                    torch.save(dict(gradients={n:v.grad.detach().cpu() for n,v in model.named_parameters()},loss=float(score.cpu()),
                        depth=depth,proposal=proposal,count=audit_count,window_start=p['audit_window_start']),a.output/f'PRIVATE-gradient-u{update}-k{depth}.pt')
            opt.zero_grad(set_to_none=True)
        dist.barrier()
    if rank==0:
        result=dict(status='COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS',world=world,arm=a.arm,updates=len(schedule),
            real_targets=exposure,nominal_training_matrix_flops=work,checkpoints=checkpoints,
            initial_model_digest=initial_digest,shared_initial_model_digest=shared_initial_digest,parameters=parameter_count,wall_seconds=time.time()-started,
            mean_warm_update_seconds=sum(r['seconds'] for r in telemetry[3:])/len(telemetry[3:]),
            all_replica_full_states_equal=True,heldout_reads=0,
            final_lr=lr,scope='Initial-learning screen from random initialization with complete independent endpoint cooldown. No teacher,auxiliary objective,loss weighting,learned allocation or convergence/strong-frontier claim.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--arm',required=True);main(p.parse_args())
