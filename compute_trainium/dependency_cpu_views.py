"""Shared-prefix CPU execution of the frozen zero-update dependency interventions."""
import torch


def dependency_views(model,view,cos_table,sin_table,*,mode,pause_id,variants):
    tokens=view['tokens'];anchors=view['anchors'];cos=cos_table.index_select(-2,view['positions']);sin=sin_table.index_select(-2,view['positions'])
    embedding=model.model.embed_tokens(tokens);x=embedding;caches=[]
    for layer in model.model.layers:
        x,k,v=layer(x,cos,sin,None,None,None,True);caches.append((k,v))
    initial=model.model.norm(x).index_select(1,anchors);raw=model.lm_head(initial);source_embedding=embedding.index_select(1,anchors);result={}
    for variant in variants:
        if variant not in ['original','strict','strict_rms','embedding_replay']:raise ValueError('Frozen intervention required')
        if variant=='embedding_replay':x=source_embedding
        elif mode=='constant':x=model.model.norm(model.model.embed_tokens(torch.full_like(view['targets'],pause_id)))
        elif mode=='latent':x=initial.clone()
        else:raise ValueError('Frozen latent or trained constant mode required')
        if variant=='strict_rms':x=x*(source_embedding.float().square().mean(-1,keepdim=True).sqrt()/x.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)).to(x.dtype)
        positions=torch.arange(tokens.shape[1]);prefix=positions[None,:]<=anchors[:,None] if variant=='original' else positions[None,:]<anchors[:,None]
        mask=torch.cat([prefix,anchors[None,:]==anchors[:,None]],1);qc=cos.index_select(-2,anchors);qs=sin.index_select(-2,anchors)
        for layer,(k,v) in zip(model.model.layers,caches,strict=True):x=layer(x,qc,qs,mask,k,v,False)
        state=model.model.norm(x);result[variant]=dict(raw_logits=raw,mixed_logits=model.lm_head(.5*initial+.5*state),initial_workspace=initial,workspace_state=state)
    return result
