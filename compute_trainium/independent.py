"""Reference CPU implementation with separate attention projections and RoPE.

This deliberately uses the standalone model.py blocks, not TorchTitan or the
native compiler. It scores the same pre-norm ordinary/recurrent architecture.
"""
import torch

from .model import LanguageModel, ModelConfig


class IndependentModel(LanguageModel):
    def __init__(self,config,repetitions=1,prelude=2,core=16,coda=2):
        super().__init__(config)
        if prelude+core+coda!=config.layers:raise ValueError('Layer partition differs')
        self.repetitions,self.prelude,self.core=repetitions,prelude,core

    def load_titan_state(self,state):
        used=set()
        def copy(destination,key):
            source=state[key]
            if destination.shape!=source.shape:raise ValueError('Weight shape differs: '+key)
            destination.copy_(source);used.add(key)
        with torch.no_grad():
            copy(self.embedding.weight,'decoder.tok_embeddings.weight')
            if not torch.equal(state['decoder.tok_embeddings.weight'],state['decoder.lm_head.weight']):
                raise ValueError('Expected tied head weights')
            used.add('decoder.lm_head.weight');copy(self.norm.weight,'decoder.norm.weight')
            for i,block in enumerate(self.blocks):
                prefix=f'decoder.layers.{i}.'
                for destination,key in [(block.q.weight,'attention.qkv_linear.wq.weight'),
                    (block.k.weight,'attention.qkv_linear.wk.weight'),(block.v.weight,'attention.qkv_linear.wv.weight'),
                    (block.o.weight,'attention.wo.weight'),(block.gate.weight,'feed_forward.w1.weight'),
                    (block.down.weight,'feed_forward.w2.weight'),(block.up.weight,'feed_forward.w3.weight'),
                    (block.attn_norm.weight,'attention_norm.weight'),(block.ffn_norm.weight,'ffn_norm.weight')]:
                    copy(destination,prefix+key)
        if used!=set(state):raise ValueError('Unmapped model state keys: '+str(set(state)-used))

    def hidden(self,tokens,attention_mask=None):
        n=tokens.shape[1]
        if n>self.config.max_seq:raise ValueError('Rotary cache exceeded')
        e=self.embedding(tokens)
        cos,sin=self.cos[:,:,:n].to(e.dtype),self.sin[:,:,:n].to(e.dtype)
        for i in range(self.prelude):e=self.blocks[i](e,cos,sin,attention_mask)
        h=e
        for iteration in range(self.repetitions):
            if iteration:h=h+e
            for i in range(self.prelude,self.prelude+self.core):
                h=self.blocks[i](h,cos,sin,attention_mask)
        for i in range(self.prelude+self.core,self.config.layers):h=self.blocks[i](h,cos,sin,attention_mask)
        return self.norm(h)
