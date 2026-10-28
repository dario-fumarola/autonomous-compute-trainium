"""Fixed-budget sparse adaptation of token/latent-step causal masking.

One continuous step,shared position IDs and differentiable prefix KV reuse.
No router,teacher,vocabulary sampling or Jacobi approximation.Reference:
https://arxiv.org/html/2602.08220v2 equations2 and8;this is NOT its full method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .workspace_olmo_candidate import Attention,Decoder,Norm,Projection,WorkspaceOlmoCandidate


class CachedAttention(Attention):
    def forward(self,x,cos,sin,mask=None,past_key=None,past_value=None,return_cache=False):
        b,n,_=x.shape
        q=self.q_norm(self.q_proj(x)).view(b,n,self.heads,self.dim).transpose(1,2)
        k=self.k_norm(self.k_proj(x)).view(b,n,self.kv,self.dim).transpose(1,2)
        v=self.v_proj(x).view(b,n,self.kv,self.dim).transpose(1,2)
        def rotate(t):
            left,right=t.chunk(2,dim=-1)
            return (t.float()*cos+torch.cat([-right.float(),left.float()],dim=-1)*sin).to(t.dtype)
        q,k=rotate(q),rotate(k)
        if self.kv!=self.heads:
            k=k.repeat_interleave(self.heads//self.kv,dim=1);v=v.repeat_interleave(self.heads//self.kv,dim=1)
        current_k,current_v=k,v
        if past_key is not None:
            if past_value is None:raise ValueError('Complete causal KV pair required')
            k=torch.cat([past_key,k],2);v=torch.cat([past_value,v],2)
        if mask is None:mask=torch.ones(n,n,dtype=torch.bool,device=x.device).tril()
        scores=(q.float()@k.float().transpose(-2,-1))*(self.dim**-.5)
        probability=torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)
        attended=(probability@v.float()).to(v.dtype).transpose(1,2).reshape(b,n,-1)
        result=self.o_proj(attended)
        return (result,current_k,current_v) if return_cache else result


class CachedDecoder(Decoder):
    def __init__(self,c,fp16):
        nn.Module.__init__(self)
        self.self_attn=CachedAttention(c,fp16)
        from .workspace_olmo_candidate import MLP
        self.mlp=MLP(c,fp16)
        self.post_attention_layernorm=Norm(c.hidden_size,c.rms_norm_eps)
        self.post_feedforward_layernorm=Norm(c.hidden_size,c.rms_norm_eps)
    def forward(self,x,cos,sin,mask=None,past_key=None,past_value=None,return_cache=False):
        result=self.self_attn(x,cos,sin,mask,past_key,past_value,return_cache)
        if return_cache:attended,k,v=result
        else:attended=result
        x=x+self.post_attention_layernorm(attended)
        x=x+self.post_feedforward_layernorm(self.mlp(x))
        return (x,k,v) if return_cache else x


def sparse_step_mask(sequence_length,anchors,*,device=None):
    if anchors.ndim!=1 or not anchors.numel() or anchors.dtype!=torch.long:raise ValueError('Nonempty int64 anchor vector required')
    # Values are runtime tensor inputs;the registered sparse schedule is fixed.
    positions=torch.arange(sequence_length,device=anchors.device)
    result=torch.cat([positions[None,:]<=anchors[:,None],anchors[None,:]<=anchors[:,None]],1)
    return result.to(device) if device is not None else result


class SparseWorkspaceCandidate(WorkspaceOlmoCandidate):
    def __init__(self,c,*,fp16_projections=False,activation_checkpointing=False):
        nn.Module.__init__(self);self.config=c;self.activation_checkpointing=activation_checkpointing
        self.model=nn.Module();self.model.embed_tokens=nn.Embedding(c.vocab_size,c.hidden_size,padding_idx=c.pad_token_id)
        self.model.layers=nn.ModuleList([CachedDecoder(c,fp16_projections) for _ in range(c.num_hidden_layers)])
        self.model.norm=Norm(c.hidden_size,c.rms_norm_eps);self.lm_head=Projection(c.hidden_size,c.vocab_size,fp16_projections)
    def workspace_mask(self,sequence_length,anchors):
        return sparse_step_mask(sequence_length,anchors)
    def sparse_forward(self,tokens,cos,sin,anchors,*,raw_positions=None,mode='latent',mix=.5,pause_id=None):
        if mode not in ['latent','detached','constant'] or not 0<=mix<=1:raise ValueError('Declared workspace mode and convex mixing required')
        x=self.model.embed_tokens(tokens);caches=[]
        for layer in self.model.layers:
            args=(x,cos,sin,None,None,None,True)
            result=checkpoint(layer,*args,use_reentrant=False,preserve_rng_state=False) if self.activation_checkpointing and torch.is_grad_enabled() else layer(*args)
            x,k,v=result;caches.append((k,v))
        base=self.model.norm(x);initial=base.index_select(1,anchors)
        raw_logits=self.lm_head(base if raw_positions is None else base.index_select(1,raw_positions))
        if mix==0:
            return dict(raw_logits=raw_logits,mixed_logits=self.lm_head(initial),initial_workspace=initial,workspace_input=None,workspace_state=None)
        if mode=='constant':
            if pause_id is None:raise ValueError('Existing nonpadding constant query ID required')
            x=self.model.norm(self.model.embed_tokens(torch.full_like(tokens[:,:anchors.numel()],pause_id)))
        else:x=initial.detach() if mode=='detached' else initial.clone()
        workspace_input=x
        query_cos=cos.index_select(-2,anchors);query_sin=sin.index_select(-2,anchors)
        mask=self.workspace_mask(tokens.shape[1],anchors)
        for layer,(k,v) in zip(self.model.layers,caches,strict=True):
            args=(x,query_cos,query_sin,mask,k,v,False)
            x=checkpoint(layer,*args,use_reentrant=False,preserve_rng_state=False) if self.activation_checkpointing and torch.is_grad_enabled() else layer(*args)
        state=self.model.norm(x);mixed=(1-mix)*initial+mix*state
        return dict(raw_logits=raw_logits,mixed_logits=self.lm_head(mixed),initial_workspace=initial,workspace_input=workspace_input,workspace_state=state)


class IndependentSparseWorkspaceCandidate(SparseWorkspaceCandidate):
    """A query sees the ordinary prefix and its own latent key only.

    Removing another token's workspace therefore leaves its prediction
    unchanged.This deliberate restriction differs from the published2D mask.
    """
    def workspace_mask(self,sequence_length,anchors):
        positions=torch.arange(sequence_length,device=anchors.device)
        return torch.cat([positions[None,:]<=anchors[:,None],anchors[None,:]==anchors[:,None]],1)
