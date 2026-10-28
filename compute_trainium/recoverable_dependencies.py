"""Original-source targets and position-preserving context interventions."""
import hashlib
import torch
import torch.nn.functional as F
from .stable_target_loss import stable_target_log_probs


def source_target(text,start,end,tokenizer,*,kind,identity,group,prefix_cap,short_tokens,antecedent_start):
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids,offsets=encoded['input_ids'],encoded['offset_mapping']
    touched=[i for i,(a,b) in enumerate(offsets) if b>start and a<end]
    if not touched:raise ValueError('empty_target')
    lo,hi=touched[0],touched[-1]+1;a,b=offsets[lo][0],offsets[hi-1][1]
    if a>start or b<end or text[a:start].strip() or text[end:b].strip():raise ValueError('target_token_edge')
    if not 1<=hi-lo<=16:raise ValueError('target_length')
    full_start=max(0,lo-prefix_cap);prefix=ids[full_start:lo]
    if len(prefix)<32 or len(prefix)<=short_tokens:raise ValueError('insufficient_prefix')
    if antecedent_start<offsets[full_start][0] or antecedent_start>=a:raise ValueError('antecedent_missing_from_full')
    short_start=lo-short_tokens
    if antecedent_start>=offsets[short_start][0]:raise ValueError('antecedent_retained_in_short')
    key=hashlib.sha256(f'{identity}:{kind}:{start}:{end}'.encode()).hexdigest()
    return dict(id=key,parent_id=identity,group=group,kind=kind,cohort=int(group,16)%2,full_prefix=prefix,short_prefix=ids[short_start:lo],answer=ids[lo:hi],original_target_span=[a,b],full_source_token_start=full_start,target_source_token_start=lo,short_offset=len(prefix)-short_tokens,source_utf8_sha256=hashlib.sha256(text.encode()).hexdigest())


def pack_dependency(case,condition,pad_id,*,positions=1024,slots=16):
    if condition not in ['full','short']:raise ValueError('Frozen original context intervention required')
    prefix=case[condition+'_prefix'];answer=case['answer'];offset=0 if condition=='full' else case['short_offset']
    if not prefix or not 1<=len(answer)<=slots or len(prefix)+slots-1>positions:raise ValueError('Bounded source/target geometry required')
    inputs=prefix+answer[:-1];inputs+= [pad_id]*(positions-len(inputs))
    anchors=torch.arange(slots)+len(prefix)-1;labels=answer+[pad_id]*(slots-len(answer));mask=torch.arange(slots)<len(answer)
    return dict(tokens=torch.tensor(inputs).reshape(1,-1),anchors=anchors,targets=torch.tensor(labels).reshape(1,-1),mask=mask.reshape(1,-1),positions=torch.arange(positions)+offset)


def dependency_metrics(raw,worked,targets,mask,initial,state):
    if raw.shape!=worked.shape or raw.shape[:-1]!=targets.shape or mask.shape!=targets.shape or not bool(mask.any()):raise ValueError('Exact real-source mask and logit geometry required')
    w=mask.to(raw.dtype);count=w.sum();rn=-stable_target_log_probs(raw,targets);zn=-stable_target_log_probs(worked,targets)
    lp=F.log_softmax(raw,-1);lq=F.log_softmax(worked,-1);rp=lp.exp();zp=lq.exp()
    rcorrect=raw.argmax(-1).eq(targets);zcorrect=worked.argmax(-1).eq(targets)
    cosine=F.cosine_similarity(initial.float(),state.float(),dim=-1);relative=(state.float()-initial.float()).square().sum(-1).sqrt()/initial.float().square().sum(-1).sqrt().clamp_min(1e-20)
    average=lambda x:(x*w).sum()/count
    return dict(raw_nll=average(rn),workspace_nll=average(zn),raw_first_nll=rn[:,0].mean(),workspace_first_nll=zn[:,0].mean(),raw_token_accuracy=average(rcorrect.to(w.dtype)),workspace_token_accuracy=average(zcorrect.to(w.dtype)),raw_exact=(rcorrect|~mask).all(-1).to(w.dtype).mean(),workspace_exact=(zcorrect|~mask).all(-1).to(w.dtype).mean(),beneficial_flip_rate=average((~rcorrect&zcorrect).to(w.dtype)),harmful_flip_rate=average((rcorrect&~zcorrect).to(w.dtype)),raw_entropy=average(-(rp*lp).sum(-1)),workspace_entropy=average(-(zp*lq).sum(-1)),raw_to_workspace_kl=average((rp*(lp-lq)).sum(-1)),state_cosine=average(cosine),state_relative_l2=average(relative),raw_state_rms=average(initial.float().square().mean(-1).sqrt()),workspace_state_rms=average(state.float().square().mean(-1).sqrt()),real_targets=count)
