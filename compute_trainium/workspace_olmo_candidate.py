"""OLMo2 forward candidate for the existing Native full-parameter sharder.

Matches the public OLMo2 architecture (Transformers5.15.1 reference,Apache2.0).
No SDK SDPA backward,no hidden caches;rotary tensors are explicit inputs.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class OlmoWorkspaceConfig:
    hidden_size:int=2048
    intermediate_size:int=8192
    num_hidden_layers:int=16
    num_attention_heads:int=16
    num_key_value_heads:int=16
    vocab_size:int=100352
    rms_norm_eps:float=1e-6
    rope_theta:float=500000.
    pad_token_id:int|None=100277

    @classmethod
    def from_hf(cls,config):
        values={k:getattr(config,k) for k in cls.__dataclass_fields__ if hasattr(config,k)}
        rope=getattr(config,'rope_parameters',None) or {}
        values['rope_theta']=rope.get('rope_theta',values.get('rope_theta',500000.))
        return cls(**values)

    def __post_init__(self):
        if (self.hidden_size%self.num_attention_heads or self.num_attention_heads%self.num_key_value_heads
                or (self.hidden_size//self.num_attention_heads)%2 or self.num_hidden_layers<1):
            raise ValueError('Valid OLMo2 head dimensions and depth required')


def rotary_tensors(config,length,device='cpu'):
    dim=config.hidden_size//config.num_attention_heads
    inv=1./(config.rope_theta**(torch.arange(0,dim,2,device=device,dtype=torch.float32)/dim))
    positions=torch.arange(length,device=device,dtype=torch.float32)
    frequency=positions[:,None]*inv[None,:]
    phase=torch.cat([frequency,frequency],dim=-1)[None,None,:,:]
    return phase.cos(),phase.sin()


class Norm(nn.Module):
    def __init__(self,size,eps):
        super().__init__();self.weight=nn.Parameter(torch.ones(size));self.eps=eps
    def forward(self,x):
        dtype=x.dtype;y=x.float();y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.eps)
        return (self.weight*y).to(dtype)


class Projection(nn.Linear):
    def __init__(self,inputs,outputs,fp16):
        super().__init__(inputs,outputs,bias=False);self.fp16=fp16
    def forward(self,x):
        dtype=torch.float16 if self.fp16 else torch.float32
        return F.linear(x.to(dtype),self.weight.to(dtype))


class Attention(nn.Module):
    def __init__(self,c,fp16):
        super().__init__();self.heads=c.num_attention_heads;self.kv=c.num_key_value_heads
        self.dim=c.hidden_size//self.heads;kv_dim=self.kv*self.dim
        self.q_proj=Projection(c.hidden_size,c.hidden_size,fp16)
        self.k_proj=Projection(c.hidden_size,kv_dim,fp16)
        self.v_proj=Projection(c.hidden_size,kv_dim,fp16)
        self.o_proj=Projection(c.hidden_size,c.hidden_size,fp16)
        self.q_norm=Norm(c.hidden_size,c.rms_norm_eps);self.k_norm=Norm(kv_dim,c.rms_norm_eps)
    def forward(self,x,cos,sin):
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
        scores=(q.float()@k.float().transpose(-2,-1))*(self.dim**-.5)
        keep=torch.ones(n,n,dtype=torch.bool,device=x.device).tril()
        probability=torch.softmax(scores.masked_fill(~keep,float('-inf')),dim=-1)
        attended=(probability@v.float()).to(v.dtype).transpose(1,2).reshape(b,n,-1)
        return self.o_proj(attended)


class MLP(nn.Module):
    def __init__(self,c,fp16):
        super().__init__();self.gate_proj=Projection(c.hidden_size,c.intermediate_size,fp16)
        self.up_proj=Projection(c.hidden_size,c.intermediate_size,fp16)
        self.down_proj=Projection(c.intermediate_size,c.hidden_size,fp16)
    def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class Decoder(nn.Module):
    def __init__(self,c,fp16):
        super().__init__();self.self_attn=Attention(c,fp16);self.mlp=MLP(c,fp16)
        self.post_attention_layernorm=Norm(c.hidden_size,c.rms_norm_eps)
        self.post_feedforward_layernorm=Norm(c.hidden_size,c.rms_norm_eps)
    def forward(self,x,cos,sin):
        x=x+self.post_attention_layernorm(self.self_attn(x,cos,sin))
        return x+self.post_feedforward_layernorm(self.mlp(x))


class WorkspaceOlmoCandidate(nn.Module):
    def __init__(self,c,*,fp16_projections=False,activation_checkpointing=False):
        super().__init__();self.config=c;self.activation_checkpointing=activation_checkpointing
        self.model=nn.Module();self.model.embed_tokens=nn.Embedding(c.vocab_size,c.hidden_size,padding_idx=c.pad_token_id)
        self.model.layers=nn.ModuleList([Decoder(c,fp16_projections) for _ in range(c.num_hidden_layers)])
        self.model.norm=Norm(c.hidden_size,c.rms_norm_eps)
        self.lm_head=Projection(c.hidden_size,c.vocab_size,fp16_projections)
    def forward(self,tokens,cos,sin,positions=None):
        x=self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            if self.activation_checkpointing and torch.is_grad_enabled():
                x=checkpoint(layer,x,cos,sin,use_reentrant=False,preserve_rng_state=False)
            else:x=layer(x,cos,sin)
        x=self.model.norm(x)
        if positions is not None:x=x.index_select(1,positions)
        return self.lm_head(x).float()
    def compile_stages(self,backend):
        options=dict(backend=backend,fullgraph=True,dynamic=False)
        self.model.embed_tokens.compile(**options)
        for layer in self.model.layers:layer.compile(**options)
        self.model.norm.compile(**options);self.lm_head.compile(**options)
