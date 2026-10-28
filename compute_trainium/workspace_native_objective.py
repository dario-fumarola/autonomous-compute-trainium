"""Fixed-shape selective-head workspace execution;full costs remain visible."""
import ctypes
import gc
import time
import torch
from .workspace_joint_terms import padded_prediction_inputs,workspace_joint_terms
from .workspace_learning import identity_reader_context
from .workspace_olmo_candidate import rotary_tensors
from .stable_target_loss import stable_target_log_probs


def drain_runtime(device):
    if torch.device(device).type=='neuron':
        gc.collect();ctypes.CDLL(None).malloc_trim(0);torch.neuron.empty_cache()


class SelectiveWorkspacePredictor:
    def __init__(self,model,*,positions=192):
        self.model=model;self.positions=positions;self.device=next(model.parameters()).device
        self.cos,self.sin=rotary_tensors(model.config,positions,self.device);self.events=[]
    def logits(self,context,targets,phase):
        started=time.perf_counter()
        inputs,where=padded_prediction_inputs(context,targets,self.positions,self.model.config.pad_token_id)
        logits=self.model(inputs.to(self.device),self.cos,self.sin,where.to(self.device))
        self.events.append(dict(phase=phase,padded_positions=self.positions,input_positions=context.shape[1]+targets.shape[1]-1,
            projected_positions=targets.shape[1],batch=context.shape[0],dispatch_wall_seconds=time.perf_counter()-started))
        return logits.float()
    def log_probs(self,context,targets,phase):
        logits=self.logits(context,targets,phase)
        return stable_target_log_probs(logits,targets.to(self.device))
    @torch.no_grad()
    def sample(self,prefix,*,format,length,group_size,generator,greedy=False):
        # The writer takes no continuation/target argument.
        if prefix.shape[0]!=1 or not 0<length<=32 or not 1<=group_size<=8:raise ValueError('Bounded single-anchor workspace sampling')
        opening=prefix.new_tensor(format.opening).reshape(1,-1)
        context=torch.cat([prefix,opening],1).expand(group_size,-1);draws=[]
        for _ in range(length):
            logits=self.logits(context,prefix.new_zeros((group_size,1)),'workspace_sampling')
            probabilities=torch.softmax(logits[:,-1,:].detach().cpu(),-1)
            token=probabilities.argmax(-1,keepdim=True) if greedy else torch.multinomial(probabilities,1,generator=generator)
            draws.append(token);context=torch.cat([context,token],1)
            del logits,probabilities;drain_runtime(self.device)
        return torch.cat(draws,1)
    @torch.no_grad()
    def score(self,prefix,targets,workspace,*,format,cost_per_token=.001):
        raw=float(self.log_probs(prefix,targets,'scorer_raw').mean().cpu());worked=[]
        drain_runtime(self.device)
        for row in workspace.split(1):
            worked.append(float(self.log_probs(identity_reader_context(prefix,row,format),targets,'scorer_workspace').mean().cpu()))
            drain_runtime(self.device)
        gain=torch.tensor(worked,dtype=torch.float32)-raw
        return dict(absolute_gain=gain,rewards=gain-cost_per_token*workspace.shape[1],raw_log_probability=raw)
    def backward(self,prefix,targets,workspace,rewards,*,format,accumulator,world,scale=256):
        values={}
        for phase,context,target,weight,reduction in workspace_joint_terms(prefix,targets,workspace,rewards,format):
            nll=-self.log_probs(context,target,phase)
            coefficient=torch.tensor(weight,dtype=torch.float32).to(self.device)
            loss=coefficient*(nll.mean() if reduction=='mean' else nll.sum())
            values[phase]=values.get(phase,0.)+float(loss.detach().cpu())
            (loss/world*scale).backward();accumulator.add_and_clear()
            del nll,loss,coefficient;drain_runtime(self.device)
        accumulator.install()
        return values
    def ledger(self):
        return dict(forward_calls=len(self.events),processed_positions=sum(e['batch']*e['padded_positions'] for e in self.events),
            projected_positions=sum(e['batch']*e['projected_positions'] for e in self.events),
            phases={p:sum(e['phase']==p for e in self.events) for p in sorted({e['phase'] for e in self.events})},
            note='Actual padded work and selective heads. Dispatch timing is asynchronous;stage drains/wall times separately measure completed execution. Positions are not FLOPs.')
