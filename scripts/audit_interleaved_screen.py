"""Independent global gradients and complete CPU state restoration, before scoring."""
import argparse,json,resource,shutil,time
from pathlib import Path


def main(a):
    import torch
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_model_factory import make_model as probability_model
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources as ProbabilitySources
    from compute_trainium.pretraining_optimizer import parameter_groups
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.training import file_sha256
    from compute_trainium.interleaved_objective import reference_loss_sum
    from scripts.qualify_native import digest
    from scripts.qualify_trained_gradient import compare_gradients
    from compute_trainium.probability_model_factory import gradient_families
    torch.set_num_threads(32);start=time.time();p=json.loads(a.plan.read_text())
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen audit changed')
    a.output.mkdir(exist_ok=False);data=ProbabilitySources(a.root/p['data_relative']);arm=p['arms'][a.arm]
    folder=a.root/'runs'/arm['phase'];rows=[]
    for update in arm['checkpoint_updates']:
        checkpoint=folder/f'PRIVATE-u{update}.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model=probability_model(p);model.load_state_dict(saved['model'])
        opt=torch.optim.AdamW(parameter_groups(model,model.embedding.weight,decay=p['weight_decay']),lr=arm['lr'],betas=(.9,.95),eps=1e-8,foreach=False,fused=True)
        opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        # Native checkpoint serialization recursively converts OrderedDict to
        # dict. Apply exactly that same representation before hashing restore;
        # tensor bytes,scalar values and key order remain strict.
        restored=digest(cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())))
        if restored!=saved['digest']:raise AssertionError('Exact model/optimizer/RNG state restoration failed')
        for depth in arm['audit_depths']:
            model.zero_grad(set_to_none=True);loss_sum=0.
            native=torch.load(folder/f'PRIVATE-gradient-u{update}-k{depth}.pt',weights_only=True,mmap=True)
            count=data.target_count(p['audit_window_start'],64*p['batch'])
            for rank in range(64):
                begin=p['audit_window_start']+rank*p['batch'];x,y,mask=data.batch(list(range(begin,begin+p['batch'])))
                value=reference_loss_sum(model,x,y,mask,depth,mode=arm['workspace_mode'])/count
                value.backward();loss_sum+=float(value.detach());del value
                if rank%8==0:(a.output/'progress.json').write_text(json.dumps(dict(update=update,depth=depth,rank=rank,seconds=time.time()-start)))
                if (time.time()-start>p['cpu_phase_seconds_limit'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3
                    or shutil.disk_usage(a.root).free<35*1024**3):raise RuntimeError('CPU audit resource bound')
            expected={n:v.grad for n,v in model.named_parameters()};actual=native['gradients'];checks={'overall':compare_gradients(actual,expected)}
            for family,match in gradient_families(p):
                checks[family]=compare_gradients({n:v for n,v in actual.items() if match(n)},{n:v for n,v in expected.items() if match(n)})
            row=dict(update=update,depth=depth,gradient=checks,loss_error=loss_sum-native['loss'],cpu_loss=loss_sum,
                native_loss=native['loss'],real_targets=count,full_state_restore_exact=True,checkpoint_sha256=file_sha256(checkpoint))
            rows.append(row);(a.output/'checks.json').write_text(json.dumps(rows,indent=2))
            if max(v['relative_l2'] for v in checks.values())>.02 or abs(row['loss_error'])>.003:raise AssertionError('Independent mature global numerical gate')
        del saved,opt,model,native,expected,actual
    result=dict(status='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS',arm=a.arm,rows=rows,assigned_chips=0,optimizer_updates=0,
        wall_seconds=time.time()-start,heldout_reads=0,scope='Complete CPU state restoration and independent SDPA/CE full-global gradients. Separate Native next-update replay was qualified on disposable weights; no stronger scientific restart equivalence claimed here.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--arm',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
