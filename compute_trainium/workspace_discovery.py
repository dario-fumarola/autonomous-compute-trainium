"""Discovery conditions and explicit, head-inclusive nominal matrix accounting.

No target enters the writer's sampling API. Draft labels supervise its writer;
the draft reader always gets the writer's own sampled tokens.
"""
import math
import torch
from .workspace_joint_terms import workspace_joint_terms
from .workspace_learning import identity_reader_context

ARMS=('ordinary','workspace','pause','draft')


def donor_indices(groups,domains,*,world=64):
    if len(groups)!=len(domains) or not groups or len(groups)%world:raise ValueError('Complete source-only evaluation batches required')
    result=[]
    for k in range(len(groups)):
        lo=k//world*world
        choices=[lo+(k-lo+offset)%world for offset in range(1,world)]
        eligible=[j for j in choices if groups[j]!=groups[k] and domains[j]==domains[k]]
        if not eligible:raise ValueError('No different-group same-domain donor;do not silently weaken control')
        result.append(eligible[0])
    return result


def discovery_terms(arm,prefix,continuation,workspace,rewards,format,*,pause_id):
    if arm not in ARMS:raise ValueError('Unknown discovery condition')
    if prefix.shape[0]!=1 or continuation.shape[0]!=1 or continuation.shape[1]<8:
        raise ValueError('One authentic source view with eight continuation tokens required')
    y=continuation[:,:4]
    if arm=='ordinary':
        yield 'ordinary_ntp',prefix[:,:1],torch.cat([prefix[:,1:],continuation],1),1.,'mean'
        return
    if arm=='workspace':
        yield from workspace_joint_terms(prefix,y,workspace,rewards,format)
        return
    yield 'ordinary_ntp',prefix[:,:1],torch.cat([prefix[:,1:],y],1),.5,'mean'
    if arm=='pause':
        # Pauses FOLLOW the prefix so their hidden states can compute from it.
        # Never use nn.Embedding's padding row as a trainable pause token.
        pauses=prefix.new_full((1,8),pause_id)
        yield 'pause_reader',torch.cat([prefix,pauses],1),y,.5,'mean'
    else:
        if workspace.shape!=(4,8):raise ValueError('Four own sampled eight-token drafts required')
        for row in workspace.split(1):
            yield 'draft_reader',identity_reader_context(prefix,row,format),y,.5/4,'mean'
        opening=prefix.new_tensor(format.opening).reshape(1,-1)
        yield 'draft_writer',torch.cat([prefix,opening],1),continuation[:,:8],.1,'mean'


def backward_terms(predictor,terms,*,accumulator,world,scale=256):
    values={}
    for phase,context,targets,weight,reduction in terms:
        nll=-predictor.log_probs(context,targets,phase)
        # Match the qualified online path: scalar values enter Native as
        # runtime tensor inputs, not device-factory Python constants.
        coefficient=torch.tensor(weight,dtype=torch.float32).to(predictor.device)
        loss=coefficient*(nll.mean() if reduction=='mean' else nll.sum())
        values[phase]=values.get(phase,0.)+float(loss.detach().cpu())
        (loss/world*scale).backward();accumulator.add_and_clear()
        del loss,nll,coefficient
        from .workspace_native_objective import drain_runtime
        drain_runtime(predictor.device)
    accumulator.install()
    return values


def forward_matrix_work(config,*,batch,positions,projected):
    """Executed dense + full-square primitive-attention operations, FMA=2.

    Includes padded positions and both attention products. This implementation
    executes full causal-mask matrices; do not halve them as a triangular kernel.
    Lookup, normalization, softmax/loss, collectives, AdamW and EMA are NOT matrix
    products and are reported in measured whole time, not silently called FLOPs.
    """
    if min(batch,positions,projected)<1 or projected>positions:raise ValueError('Valid executed geometry required')
    d=config.hidden_size;m=config.intermediate_size;l=config.num_hidden_layers
    kv=d*config.num_key_value_heads//config.num_attention_heads
    body=2*batch*positions*l*(2*d*d+2*d*kv+3*d*m)
    attention=4*batch*l*positions*positions*d
    head=2*batch*projected*d*config.vocab_size
    return dict(body=body+attention,head=head)


def step_matrix_work(config,arm):
    if arm not in ARMS:raise ValueError('Unknown discovery condition')
    # Activation-checkpointed training: forward + block recomputation + 2x
    # backward body; the selected vocabulary head is not checkpointed (3x).
    def cost(batch,positions,projected,training=False):
        work=forward_matrix_work(config,batch=batch,positions=positions,projected=projected)
        return (4 if training else 1)*work['body']+(3 if training else 1)*work['head']
    if arm=='ordinary':return cost(1,160,159,True)
    total=cost(1,192,131,True)
    if arm=='pause':return total+cost(1,192,4,True)
    total+=8*cost(4,192,1)+4*cost(1,192,4,True)
    if arm=='draft':return total+cost(1,192,8,True)
    return total+cost(1,192,4)+4*cost(1,192,4)+4*cost(1,192,8,True)


def matched_updates(config,*,workspace_updates=48):
    if type(workspace_updates) is not int or workspace_updates<1:raise ValueError('Fixed positive pilot budget required')
    budget=workspace_updates*step_matrix_work(config,'workspace')
    # All controls are rounded UP: a fractional step must not favor workspace.
    return {arm:math.ceil(budget/step_matrix_work(config,arm)) for arm in ARMS}
