"""Disposable clipping/AdamW comparisons on saved mature TRAIN gradients."""
import argparse
import copy
import gc
import json
from pathlib import Path
import resource
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_latent_exposure_sources import guard, read
from scripts.readout_interleaved_bracket_type import checkpoint

ID='bucketed-mature-cpu-v1'
PUBLIC=Path('results/gradient_bucketing_20261030/mature-cpu-v1')


def errors(left,right):
    if set(left)!=set(right):raise AssertionError('Complete identical tensor maps required')
    numerator=denominator=maximum=0.
    for name,a in left.items():
        b=right[name]
        if a.shape!=b.shape:raise AssertionError('Shape mismatch')
        a,b=a.double(),b.double()
        if not a.isfinite().all() or not b.isfinite().all():raise AssertionError('Finite tensors required')
        numerator+=float((a-b).square().sum());denominator+=float(a.square().sum())
        maximum=max(maximum,float((a-b).abs().max()))
    return dict(relative_l2=(numerator/max(1e-30,denominator))**.5,maximum_absolute=maximum)


def run(root):
    import torch
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.bucketed_gradient_reduction import reduce_and_clip
    started=time.time();out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve each qualification attempt')
    files=sorted({str(f) for f in Path('compute_trainium').glob('*.py')} | {
           'scripts/qualify_bucketed_mature_cpu.py','scripts/audit_latent_exposure_sources.py',
           'scripts/readout_interleaved_bracket_type.py','tests/test_bucketed_gradient_reduction.py',
           'docs/BUCKETED_MATURE_CPU_20261030.md'})
    points={}
    for role in ('ordinary','pause','latent'):
        point=checkpoint(root,'latent-learning-exposure-v1-'+role,role)
        for depth in ((2,3) if role=='latent' else (0,)):
            gradient=root/'runs'/f'latent-learning-exposure-v1-{role}-fit'/f'PRIVATE-gradient-u15287-k{depth}.pt'
            points[f'{role}-k{depth}']=dict(**point,role=role,depth=depth,
                gradient_relative=str(gradient.relative_to(root)),gradient_sha256=file_sha256(gradient))
    p=dict(identity=ID,registered_unix=started,absolute_deadline_unix=started+600,threads=8,
        source_sha256={f:file_sha256(f) for f in files},points=points,multipliers=[1.,10.],
        relative_error_limit=1e-5,bucket_bytes=32*1024**2,assigned_chips=0,scientific_optimizer_updates=0,
        scope=Path('docs/BUCKETED_MATURE_CPU_20261030.md').read_text())
    out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    records={};result=dict(status='STOPPED_BUCKETED_MATURE_CPU')
    try:
        for key,point in points.items():
            guard(root,p)
            if (file_sha256(root/point['checkpoint_relative'])!=point['checkpoint_sha256']
                or file_sha256(root/point['plan_relative'])!=point['plan_sha256']
                or file_sha256(root/point['gradient_relative'])!=point['gradient_sha256']):
                raise AssertionError('Audited checkpoint or gradient changed')
            fit=read(root/point['plan_relative'])
            saved=torch.load(root/point['checkpoint_relative'],map_location='cpu',mmap=True,weights_only=False)
            captured=torch.load(root/point['gradient_relative'],map_location='cpu',mmap=True,weights_only=False)
            if captured['depth']!=point['depth']:raise AssertionError('Supported audit gradient depth required')
            for multiplier in p['multipliers']:
                models=[];optimizers=[]
                for _ in range(2):
                    m=(tied_model(fit) if point['role']=='ordinary' else make_model(fit))
                    m.load_state_dict(saved['model']);models.append(m)
                    opt=torch.optim.AdamW(parameter_groups(m,m.embedding.weight,decay=fit['weight_decay']),
                        lr=.001,betas=(.9,.95),eps=1e-8,foreach=False,fused=False)
                    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
                    for group in opt.param_groups:group['fused']=False;group['foreach']=False
                    optimizers.append(opt)
                    if set(dict(m.named_parameters()))!=set(captured['gradients']):
                        raise AssertionError('Complete mature gradient required')
                    for name,v in m.named_parameters():v.grad=captured['gradients'][name].clone()*multiplier
                a,b=models
                before={name:v.detach().clone() for name,v in a.named_parameters()}
                norm_a=torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True)
                norm_b,buckets=reduce_and_clip(b.parameters(),lambda flat:None,1,bucket_bytes=p['bucket_bytes'])
                gradient=errors({n:v.grad for n,v in a.named_parameters()},{n:v.grad for n,v in b.named_parameters()})
                optimizers[0].step();optimizers[1].step()
                delta=errors({n:v.detach()-before[n] for n,v in a.named_parameters()},
                             {n:v.detach()-before[n] for n,v in b.named_parameters()})
                moment={}
                for field in ('exp_avg','exp_avg_sq'):
                    moment[field]=errors({n:optimizers[0].state[v][field] for n,v in a.named_parameters()},
                                         {n:optimizers[1].state[v][field] for n,v in b.named_parameters()})
                own=dict(reference_norm=float(norm_a),bucketed_norm=float(norm_b),bucket_count=buckets,
                    parameter_tensor_count=len(before),gradient=gradient,parameter_update=delta,moments=moment)
                records[key+'-scale'+str(multiplier)]=own
                write_state(out/'progress.json',dict(cases=len(records),wall_seconds=time.time()-started))
                if any(v['relative_l2']>p['relative_error_limit'] for v in [gradient,delta,*moment.values()]):
                    raise AssertionError('Registered complete-gradient/update/moment parity limit')
                del models,optimizers,a,b,m,opt,v,before;gc.collect();guard(root,p)
            del saved,captured;gc.collect()
        if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):
            raise AssertionError('Frozen source changed')
        result.update(status='PASS_BUCKETED_MATURE_CPU',cases=records,native_collectives_qualified=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error),cases=records);raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.time()-started,cpu_seconds=u.ru_utime+u.ru_stime,
            peak_rss_bytes=u.ru_maxrss*1024,assigned_chips=0,scientific_optimizer_updates=0,
            disposable_cpu_optimizer_updates=2*len(records),model_scores=0,automatic_promotion=False,
            plan_sha256=file_sha256(out/'plan.json'),scope=p['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('scope','cases')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
