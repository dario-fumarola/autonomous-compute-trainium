"""Zero-update descriptive geometry on already captured TRAIN gradients."""
import argparse,json,math,time,resource
from pathlib import Path
import torch
from compute_trainium.training import file_sha256

def geometry(grads,names):
    keys=sorted(set.intersection(*(set(g) for g in grads)))
    def gram(select):
        selected=[n for n in keys if select(n)]
        out=torch.zeros((len(grads),len(grads)),dtype=torch.float64)
        for n in selected:
            vs=[g[n].detach().double().flatten() for g in grads]
            for i,a in enumerate(vs):
                for j in range(i,len(vs)):
                    v=torch.dot(a,vs[j]);out[i,j]+=v
                    if j!=i:out[j,i]+=v
        norms=out.diag().sqrt();den=norms[:,None]*norms[None,:]
        cos=torch.where(den>0,out/den,torch.zeros_like(out))
        dot=out.mean(1)
        return dict(parameters=len(selected),norms=norms.tolist(),cosines=cos.tolist(),
          dot_with_equal_depth_mean=dot.tolist(),first_order_loss_change_under_negative_mean_gradient=(-dot).tolist())
    return dict(depths=names,overall=gram(lambda n:True),embedding=gram(lambda n:n=='embedding.weight'),
       body=gram(lambda n:n not in ('embedding.weight','remaining_budget_cues')),
       cues=gram(lambda n:n=='remaining_budget_cues'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8);start=time.time()
    identity='probability-budget-cue-screen-recovery-v1'
    audit=a.root/identity/'audit-workspace/report.json'
    r=json.loads(audit.read_text())
    if r['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS' or sorted(x['depth'] for x in r['rows'])!=[0,1,2]:raise AssertionError('All endpoint audits must pass')
    phase=a.root/'runs'/(identity+'-workspace');paths=[phase/f'PRIVATE-gradient-u1530-k{k}.pt' for k in range(3)]
    hashes={f.name:file_sha256(f) for f in paths};caps=[torch.load(f,map_location='cpu',weights_only=True,mmap=True) for f in paths]
    result=geometry([x['gradients'] for x in caps],[0,1,2])
    ck=phase/'PRIVATE-u1530.pt';s=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)['model']
    cue=s['remaining_budget_cues'];e=s['embedding.weight'];result['cue_row_l2']=cue.norm(dim=1).tolist();result['embedding_row_l2_mean']=float(e.norm(dim=1).mean())
    result.update(status='COMPLETE_TRAIN_DEPTH_GRADIENT_GEOMETRY',audit_sha256=file_sha256(audit),capture_sha256=hashes,
      cpu_audit_losses=[x['cpu_loss'] for x in sorted(r['rows'],key=lambda x:x['depth'])],
      optimizer_updates=0,heldout_reads=0,assigned_chips=0,wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      scope='Descriptive unpreconditioned local gradients at one audited endpoint on identical unused TRAIN audit targets. Negative cosine means opposing first-order directions; positive cosine does not exclude optimization interference or different minima. Mean-gradient directional derivatives ignore AdamW preconditioning,decay,clipping and finite steps. No learning,selection or causal claim.')
    if result['peak_rss_bytes']>48*1024**3 or result['wall_seconds']>300:raise RuntimeError('Bound exceeded')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
