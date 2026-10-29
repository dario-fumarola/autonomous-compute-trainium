"""Reuse audited TRAIN gradients to describe agreement between supported exits."""
import argparse,json,time
from pathlib import Path

def main(root):
    import torch
    from compute_trainium.training import file_sha256
    from scripts.qualify_trained_gradient import compare_gradients
    torch.set_num_threads(8);start=time.time();out=root/'probability-exit-gradient-alignment-v1';out.mkdir(exist_ok=False)
    parent=root/'runs/probability-workspace-screen-v1-workspace'
    audit=root/'probability-workspace-screen-recovery-v1/audit-workspace/report.json'
    if json.loads(audit.read_text())['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('Audited captures required')
    paths={f'{u}-{k}':parent/f'PRIVATE-gradient-u{u}-k{k}.pt' for u in (492,1530) for k in (0,1,2)}
    plan=dict(registered_unix=start,seconds_limit=120,source_sha256=file_sha256(__file__),audit_sha256=file_sha256(audit),
        captures={k:file_sha256(f) for k,f in paths.items()},scope='Recorded TRAIN audit gradients only;no new source/model evaluation,updates,selection or promotion. Raw gradient alignment is not AdamW update alignment or a causal optimization diagnosis.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for u in (492,1530):
        loaded={k:torch.load(paths[f'{u}-{k}'],weights_only=True,map_location='cpu',mmap=True) for k in (0,1,2)}
        if len({(r['count'],r['window_start']) for r in loaded.values()})!=1:raise AssertionError('Identical TRAIN targets required')
        for left,right in ((0,1),(1,2),(0,2)):
            a=loaded[left]['gradients'];b=loaded[right]['gradients'];families={}
            for name,keep in [('all',lambda n:True),('embedding',lambda n:n=='embedding.weight'),('body',lambda n:n!='embedding.weight')]:
                comparison=compare_gradients({n:v for n,v in a.items() if keep(n)},{n:v for n,v in b.items() if keep(n)})
                families[name]={k:comparison[k] for k in ('cosine','reference_norm','candidate_norm','relative_l2')}
            rows.append(dict(update=u,left_depth=left,right_depth=right,families=families))
            if time.time()-start>120:raise RuntimeError('Bounded analysis')
    result=dict(status='COMPLETE_RECORDED_TRAIN_GRADIENT_ALIGNMENT',rows=rows,wall_seconds=time.time()-start,
        new_model_forwards=0,new_loss_scores=0,heldout_reads=0,optimizer_updates=0,assigned_chips=0,
        interpretation='Descriptive raw-gradient agreement at two saved states on one fixed TRAIN batch. Does not identify the cause of quality differences or justify changing the frozen training comparison.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
