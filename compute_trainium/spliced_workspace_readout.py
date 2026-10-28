"""Global TRAIN-fitted readouts;no per-target or held-out selection."""
import torch
from .stable_target_loss import stable_target_log_probs


def readout_grid(raw_logits,worked_logits,targets,selected,*,alphas,temperatures):
    if raw_logits.shape[1]!=32 or selected.numel()!=4 or worked_logits.shape[1]!=4:raise ValueError('Frozen32-target/four-query diagnostic required')
    if not alphas or any(not 0<=a<=.5 for a in alphas) or any(t<=0 for t in temperatures):raise ValueError('Interpolate supported endpoints only;positive temperatures')
    a=torch.tensor(alphas,dtype=raw_logits.dtype).to(raw_logits.device)
    t=torch.tensor(temperatures,dtype=raw_logits.dtype).to(raw_logits.device)
    base=raw_logits.index_select(1,selected);labels=targets.index_select(1,selected)
    raw=-stable_target_log_probs(raw_logits[None]/t[:,None,None,None],targets[None].expand(len(temperatures),-1,-1))
    mixed=base[None]+(2*a[:,None,None,None])*(worked_logits-base)[None]
    work=-stable_target_log_probs(mixed[:,None]/t[None,:,None,None,None],labels[None,None].expand(len(alphas),len(temperatures),-1,-1))
    raw_mean=raw.mean((-2,-1));raw_anchor=raw.index_select(-1,selected).mean((-2,-1));worked=work.mean((-2,-1))
    return dict(raw32=raw_mean,raw_anchor=raw_anchor,worked_anchor=worked,policy32=raw_mean[None]+.125*(worked-raw_anchor[None]))


def choose_readout(rows,*,alphas,temperatures,ordinary=False,split):
    if split!='train':raise ValueError('Selection is allowed on TRAIN only')
    if not rows or {r['domain'] for r in rows}!={0,1,2}:raise ValueError('All three natural domains required')
    def weighted(field,a=None,t=None):
        def value(r):return r[field][t] if a is None else r[field][a][t]
        return sum(w*sum(value(r) for r in rows if r['domain']==d)/sum(r['domain']==d for r in rows) for d,w in enumerate([.6,.2,.2]))
    choices=[(weighted('policy32',a,t),alphas[a],abs(temperatures[t]-1),a,t) for a in range(len(alphas)) if not ordinary or alphas[a]==0 for t in range(len(temperatures))]
    best=min(choices);raw=min((weighted('raw32',t=t),abs(temperatures[t]-1),t) for t in range(len(temperatures)))
    return dict(alpha_index=best[-2],temperature_index=best[-1],raw_temperature_index=raw[-1],alpha=alphas[best[-2]],temperature=temperatures[best[-1]],raw_temperature=temperatures[raw[-1]],train_policy32=best[0],train_calibrated_raw32=raw[0],selection_split='train',views=len(rows),scope='One global choice per model.Fixed before held-out scoring;no target-aware per-position choice.')


def source_reserve_indices(groups,*,trained_views,count,excluded_groups=()):
    if not 0<trained_views<len(groups) or count<1:raise ValueError('Positive held-back natural source allowance required')
    forbidden=set(groups[:trained_views])|set(excluded_groups)
    selected=[i for i in range(trained_views,len(groups)) if groups[i] not in forbidden][:count]
    if len(selected)!=count:raise ValueError('Insufficient source-separated calibration reserve;never reuse training groups')
    assert not ({groups[i] for i in selected}&forbidden)
    return selected
