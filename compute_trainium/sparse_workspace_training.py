"""Fixed sparse source views and the common target-emphasized objective."""
import torch
from .stable_target_loss import stable_target_log_probs


ANCHORS=(127,135,143,151)


def pack_view(view,pad_id):
    full=torch.cat([view['prefix'],view['continuation']],1)
    if full.shape[1]!=160:raise ValueError('Frozen128+32 natural source view required')
    return torch.cat([full[:,:-1],full.new_full((full.shape[0],1),pad_id)],1),full[:,1:]


def sparse_objective(model,tokens,targets,cos,sin,anchors,*,mode='latent',pause_id=None):
    raw_positions=torch.arange(159,device=tokens.device)
    if mode=='ordinary':
        logits=model(tokens,cos,sin,positions=raw_positions);mixed=logits.index_select(1,anchors)
    else:
        result=model.sparse_forward(tokens,cos,sin,anchors,raw_positions=raw_positions,
            mode=mode,mix=.5,pause_id=pause_id)
        logits,mixed=result['raw_logits'],result['mixed_logits']
    raw=-stable_target_log_probs(logits,targets).mean()
    anchor=-stable_target_log_probs(mixed,targets.index_select(1,anchors)).mean()
    return .5*raw+.5*anchor,raw,anchor


def sparse_matrix_work(config,mode='latent'):
    """Dense matrix scalar ops incl.checkpoint body recompute;not hardware time."""
    if mode not in ['latent','constant','ordinary']:raise ValueError('Registered arm required')
    d,f,l,h,kv=config.hidden_size,config.intermediate_size,config.num_hidden_layers,config.num_attention_heads,config.num_key_value_heads
    kv_width=d*kv//h;n=160;q=0 if mode=='ordinary' else 4
    dense=2*(2*d*d+2*d*kv_width+3*d*f)*(n+q)*l
    attention=4*d*(n*n+q*(n+q))*l
    # Ordinary needs only one159-row head;anchor emphasis indexes those logits.
    head_rows=159 if mode=='ordinary' else 163
    return 4*(dense+attention)+3*2*head_rows*d*config.vocab_size
