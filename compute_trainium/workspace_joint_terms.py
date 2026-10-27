"""Memory-bounded decomposition of the exact registered workspace objective."""
import torch
from .workspace_learning import group_advantages,identity_reader_context


def workspace_joint_terms(prefix,targets,workspace,rewards,format):
    if prefix.shape[0]!=1 or targets.shape[0]!=1 or workspace.shape[0]!=rewards.numel():raise ValueError('One anchor,one reward per sampled trajectory')
    group=workspace.shape[0];advantages=group_advantages(rewards)
    yield 'ordinary_ntp',prefix[:,:1],torch.cat([prefix[:,1:],targets],1),.5,'mean'
    for i in range(group):
        yield 'workspace_reader',identity_reader_context(prefix,workspace[i:i+1],format),targets,.5/group,'mean'
    opening=prefix.new_tensor(format.opening).reshape(1,-1)
    context=torch.cat([prefix,opening],1)
    for i in range(group):yield 'workspace_policy',context,workspace[i:i+1],float(advantages[i])*.1/group,'sum'


def padded_prediction_inputs(context,targets,length,pad_token_id):
    """Right padding never enters the causal states at the selected positions."""
    inputs=torch.cat([context,targets[:,:-1]],1)
    if inputs.shape[1]>length:raise ValueError('Registered fixed context allowance exceeded')
    positions=torch.arange(context.shape[1]-1,inputs.shape[1],device=inputs.device)
    padded=torch.full((inputs.shape[0],length),pad_token_id,dtype=torch.long,device=inputs.device)
    padded[:,:inputs.shape[1]]=inputs
    return padded,positions
