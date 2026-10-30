"""Independent complete first/mature gradient and AdamW checks for bucketing."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time


def main(a):
    import torch
    from compute_trainium.interleaved_objective import reference_loss_sum
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.probability_model_factory import gradient_families
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.queue import write_state
    from compute_trainium.tied_interleaved_ordinary import make_model
    from compute_trainium.training import file_sha256
    from scripts.qualify_bucketed_mature_cpu import errors
    from scripts.qualify_trained_gradient import compare_gradients
    started=time.time();torch.set_num_threads(16);torch.set_num_interop_threads(1)
    p=json.loads(a.plan.read_text())
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen plan changed')
    a.output.mkdir(exist_ok=False);parent=a.root/'runs'/p['warm_phase']
    data=ProbabilitySources(a.root/p['data_relative']);model=make_model(p)
    records={};result=dict(status='STOPPED_BUCKETED_NATIVE_CPU_AUDIT')
    def check_bound():
        if time.time()-started>p['cpu_phase_seconds_limit'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*1024**3:
            raise RuntimeError('CPU audit bound')
    def compare(actual,expected):
        out={'overall':compare_gradients(actual,expected)}
        for family,match in gradient_families(p):
            out[family]=compare_gradients({n:v for n,v in actual.items() if match(n)},{n:v for n,v in expected.items() if match(n)})
        if any(v['relative_l2']>.02 for v in out.values()):raise AssertionError('Complete gradient family gate')
        return out
    def step_comparison(before,reference,actual):
        delta=errors({n:reference['model'][n]-v for n,v in before.items()},
                     {n:actual['model'][n]-v for n,v in before.items()})
        moments={}
        if reference['optimizer']['param_groups']!=actual['optimizer']['param_groups']:
            # Only backend execution flags may differ between CPU and Native.
            groups=lambda s:[{k:v for k,v in g.items() if k not in ('fused','foreach')} for g in s['optimizer']['param_groups']]
            if groups(reference)!=groups(actual):raise AssertionError('Optimizer parameter groups differ')
        for field in ('exp_avg','exp_avg_sq'):
            moments[field]=errors({i:v[field] for i,v in reference['optimizer']['state'].items()},
                                  {i:v[field] for i,v in actual['optimizer']['state'].items()})
        for i,v in reference['optimizer']['state'].items():
            if not torch.equal(v['step'].cpu(),actual['optimizer']['state'][i]['step'].cpu()):raise AssertionError('AdamW step differs')
        if max(delta['relative_l2'],*(v['relative_l2'] for v in moments.values()))>.02:
            raise AssertionError('Complete update/moment gate')
        return dict(parameter_update=delta,moments=moments)
    try:
        initial=torch.load(parent/'PRIVATE-initial.pt',map_location='cpu',mmap=True,weights_only=True)
        for update in (1,104):
            check_bound()
            saved=None if update==1 else torch.load(parent/'PRIVATE-bucketed1-u103.pt',map_location='cpu',mmap=True,weights_only=False)
            model.load_state_dict(initial if saved is None else saved['model']);model.zero_grad(set_to_none=True)
            base=(update-1)*64*p['batch'];count=data.target_count(base,64*p['batch']);total=0.
            for rank in range(64):
                x,y,mask=data.batch(list(range(base+rank*p['batch'],base+(rank+1)*p['batch'])))
                value=reference_loss_sum(model,x,y,mask,0,mode='ordinary')/count
                value.backward();total+=float(value.detach());del value
                if rank%8==0:write_state(a.output/'progress.json',dict(update=update,rank=rank,seconds=time.time()-started))
                check_bound()
            trials=('original1','bucketed1') if update==1 else ('bucketed1',)
            own={}
            for trial in trials:
                native=torch.load(parent/f'PRIVATE-{trial}-gradient-u{update}.pt',map_location='cpu',mmap=True,weights_only=True)
                checks=compare(native['gradients'],{n:v.grad for n,v in model.named_parameters()})
                loss_error=total-native['loss']
                if abs(loss_error)>.003:raise AssertionError('Independent loss gate')
                own[trial]=dict(gradients=checks,cpu_loss=total,native_loss=native['loss'],loss_error=loss_error)
                del native
            records[str(update)]=own
            if update==1:
                before={n:v.detach().clone() for n,v in model.named_parameters()}
                opt=torch.optim.AdamW(parameter_groups(model,model.embedding.weight,decay=.1),
                    lr=p['lr'],betas=(.9,.95),eps=1e-8,foreach=False,fused=False)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                reference=dict(model=model.state_dict(),optimizer=opt.state_dict())
                for trial in trials:
                    actual=torch.load(parent/f'PRIVATE-{trial}-u1.pt',map_location='cpu',mmap=True,weights_only=False)
                    own[trial]['optimizer']=step_comparison(before,reference,actual);del actual
                original=torch.load(parent/'PRIVATE-original1-u1.pt',map_location='cpu',mmap=True,weights_only=False)
                bucketed=torch.load(parent/'PRIVATE-bucketed1-u1.pt',map_location='cpu',mmap=True,weights_only=False)
                records['native_first_update']=step_comparison(before,original,bucketed)
                ga=torch.load(parent/'PRIVATE-original1-gradient-u1.pt',map_location='cpu',mmap=True,weights_only=True)
                gb=torch.load(parent/'PRIVATE-bucketed1-gradient-u1.pt',map_location='cpu',mmap=True,weights_only=True)
                records['native_first_gradient']=compare(gb['gradients'],ga['gradients'])
                del before,opt,reference,original,bucketed,ga,gb
            del saved;gc.collect()
        result.update(status='PASS_BUCKETED_NATIVE_COMPLETE_CPU_AUDIT',checks=records)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error),checks=records);raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.time()-started,cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_bytes=u.ru_maxrss*1024,
            assigned_chips=0,scientific_optimizer_updates=0,disposable_cpu_optimizer_updates=1)
        write_state(a.output/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
