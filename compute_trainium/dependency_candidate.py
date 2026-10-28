"""Zero-update mechanism interventions;the original predictor stays intact."""
import torch
from .sparse_workspace_candidate import IndependentSparseWorkspaceCandidate


class DependencyCandidate(IndependentSparseWorkspaceCandidate):
    def dependency_forward(self,tokens,cos,sin,anchors,*,raw_positions=None,mode='latent',pause_id=None,variant='original'):
        if variant=='original':return self.sparse_forward(tokens,cos,sin,anchors,raw_positions=raw_positions,mode=mode,pause_id=pause_id,mix=.5)
        if variant not in ['strict','strict_rms','embedding_replay']:raise ValueError('Frozen one-factor input/cache intervention required')
        embedding=self.model.embed_tokens(tokens);x=embedding;caches=[]
        for layer in self.model.layers:
            x,k,v=layer(x,cos,sin,None,None,None,True);caches.append((k,v))
        base=self.model.norm(x);initial=base.index_select(1,anchors);raw=self.lm_head(base if raw_positions is None else base.index_select(1,raw_positions));source_embedding=embedding.index_select(1,anchors)
        if variant=='embedding_replay':x=source_embedding
        elif mode=='constant':x=self.model.norm(self.model.embed_tokens(torch.full_like(tokens[:,:anchors.numel()],pause_id)))
        elif mode=='latent':x=initial.clone()
        else:raise ValueError('Frozen latent or trained constant query required')
        if variant=='strict_rms':
            desired=source_embedding.float().square().mean(-1,keepdim=True).sqrt();have=x.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12);x=x*(desired/have).to(x.dtype)
        workspace_input=x;positions=torch.arange(tokens.shape[1],device=anchors.device)
        # Replace the current position's representation;do not count its
        # original KV and the new self KV twice at the same RoPE position.
        mask=torch.cat([positions[None,:]<anchors[:,None],anchors[None,:]==anchors[:,None]],1);qc=cos.index_select(-2,anchors);qs=sin.index_select(-2,anchors)
        for layer,(k,v) in zip(self.model.layers,caches,strict=True):x=layer(x,qc,qs,mask,k,v,False)
        state=self.model.norm(x);return dict(raw_logits=raw,mixed_logits=self.lm_head(.5*initial+.5*state),initial_workspace=initial,workspace_input=workspace_input,workspace_state=state)
