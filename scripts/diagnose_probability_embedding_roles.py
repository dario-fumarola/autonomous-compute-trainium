"""Frozen TRAIN-only gradient-role decomposition; zero updates and no DEV scores."""
import argparse,json,resource,time,signal
from pathlib import Path
import torch
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.probability_workspace_data import ProbabilitySources
from compute_trainium.probability_role_gradients import role_replay
from compute_trainium.stable_target_loss import stable_target_log_probs
from compute_trainium.training import file_sha256

ID='probability-embedding-roles-v1'
PUBLIC=Path('results/probability_workspace_20261029/embedding-roles-v1')

def main(root):
    start=time.time();out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);torch.set_num_threads(12)
    p=json.loads((root/'probability-fixed-one-v1/plan.json').read_text());data=ProbabilitySources(root/p['data_relative']);rows=[]
    def timeout(*args):raise TimeoutError('Registered600-second CPU diagnostic bound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    selected={'fixed_one':('workspace',1),'ordinary_data':('ordinary_data',0)};inputs={}
    for label,(arm,steps) in selected.items():
        path=root/'runs'/p['arms'][arm]['phase']/f"PRIVATE-u{p['arms'][arm]['checkpoint_updates'][-1]}.pt"
        audit=json.loads((root/p['identity']/('audit-'+arm)/'report.json').read_text())
        sha=file_sha256(path)
        if audit['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS' or sha not in {x['checkpoint_sha256'] for x in audit['rows']}:raise AssertionError('Audited immutable checkpoint required')
        inputs[label]=dict(checkpoint_relative=str(path.relative_to(root)),checkpoint_sha256=sha,extra_steps=steps)
    plan=dict(identity=ID,registered_unix=start,seconds_limit=600,cpu_threads=12,maximum_rss_bytes=48*2**30,window_start=p['audit_window_start'],windows=64*p['batch'],inputs=inputs,
      source_sha256={name:file_sha256(name) for name in ['scripts/diagnose_probability_embedding_roles.py','compute_trainium/probability_role_gradients.py']},optimizer_updates=0,heldout_reads=0,assigned_chips=0,
      scope='Same qualified global TRAIN audit batch. Separate identical leaf copies of input/output/proposal embedding roles,without changing the forward function. Their gradients must sum to the original audited tied-table gradient within the existing2% rule. Compare fixed-one with ordinary equal-data. Local cancellation is not proof that untying improves prediction and does not admit another fit.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_EMBEDDING_ROLE_DIAGNOSIS',rows=[])
    try:
        count=data.target_count(plan['window_start'],plan['windows'])
        for label,item in inputs.items():
            path=root/item['checkpoint_relative'];saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            model=ProbabilityWorkspace(ModelConfig(**p['model_config']));model.load_state_dict(saved['model']);model.requires_grad_(False)
            leaves=[model.embedding.weight.detach().clone().requires_grad_() for _ in range(3)]
            for offset in range(0,plan['windows'],p['batch']):
                x,y,mask=data.batch(list(range(plan['window_start']+offset,plan['window_start']+offset+p['batch'])))
                logits=role_replay(model,x,*leaves,item['extra_steps']);loss=-(stable_target_log_probs(logits,y)*mask).sum()/count;loss.backward();del logits,loss
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>plan['maximum_rss_bytes']:raise RuntimeError('CPU RSS bound')
            names=['input','output','proposal_value'];grads={name:leaf.grad for name,leaf in zip(names,leaves) if leaf.grad is not None}
            original=torch.load(path.parent/f"PRIVATE-gradient-u{saved['update']}-k{item['extra_steps']}.pt",map_location='cpu',weights_only=False,mmap=True)['gradients']['embedding.weight']
            total=sum(grads.values());relative=float((total-original).double().norm()/original.double().norm())
            if relative>.02:raise AssertionError('Embedding-role gradients do not reconstruct audited gradient')
            norms={name:float(g.double().norm()) for name,g in grads.items()}
            cosines={f'{a}_vs_{b}':float((grads[a].double()*grads[b].double()).sum()/(norms[a]*norms[b])) for i,a in enumerate(grads) for b in list(grads)[i+1:]}
            rows.append(dict(model=label,gradient_role_norms=norms,role_cosines=cosines,summed_gradient_norm=float(total.double().norm()),sum_of_role_norms=sum(norms.values()),norm_ratio=float(total.double().norm())/sum(norms.values()),reconstruction_relative_error=relative,real_targets=count))
            del model,leaves,grads,total,original,saved
        result.update(status='COMPLETE_EMBEDDING_ROLE_DIAGNOSIS',rows=rows)
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_updates=0,heldout_reads=0,assigned_chips=0,automatic_fit_admission=False)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
