"""Capture the rejected readout batch and compare against FP32/FP64 references."""
import argparse,json,os,time
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

def main(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import ProbabilityWorkspace
    from compute_trainium.probability_intervention import ProposalReadout
    from compute_trainium.stable_target_loss import stable_target_log_probs
    from compute_trainium.training import file_sha256
    p=json.loads(a.plan.read_text());src=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(src/f)!=h for f,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen diagnosis changed')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
    if world!=64:raise AssertionError('Original topology required')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port);torch.neuron.set_device(0);dist.init_process_group('neuron');a.output.mkdir(parents=True,exist_ok=True);start=time.time()
    saved=torch.load(a.root/p['checkpoint_relative'],map_location='cpu',weights_only=False,mmap=True)
    if file_sha256(a.root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('Exact failed checkpoint required')
    cfg=ModelConfig(**p['model_config']);m=ProbabilityWorkspace(cfg,explicit_attention=True);m.load_state_dict(saved['model']);m.to('neuron')
    permutation=torch.randperm(cfg.vocab_size,generator=torch.Generator().manual_seed(p['permutation_seed']))
    class Score(nn.Module):
        def __init__(self):super().__init__();self.reader=ProposalReadout(m,2,permutation=permutation)
        def forward(self,x,y):return -stable_target_log_probs(self.reader(x),y)
    score=torch.compile(Score().to('neuron'),backend='neuron',fullgraph=True,dynamic=False)
    tokens=np.load(a.root/p['evaluation_relative']/'broad-tokens.npy',mmap_mode='r')[128:132]
    x=torch.from_numpy(np.asarray(tokens[:,:-1],dtype=np.int64));y=torch.from_numpy(np.asarray(tokens[:,1:],dtype=np.int64))
    with torch.no_grad():actual=score(x.to('neuron'),y.to('neuron')).cpu()
    torch.save(actual,a.output/f'PRIVATE-rank{rank}-loss.pt');dist.barrier()
    if rank==0:
        torch.set_num_threads(12);references={}
        for name,explicit,dtype in [('sdpa32',False,torch.float32),('sdpa64',False,torch.float64),('explicit64',True,torch.float64)]:
            ref=ProbabilityWorkspace(cfg,explicit_attention=explicit).to(dtype);ref.load_state_dict(saved['model']);reader=ProposalReadout(ref,2,permutation=permutation)
            with torch.no_grad():logits=reader(x);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
            references[name]=loss.double();del ref,reader,logits
        error={name:float((actual.double()-value).abs().max()) for name,value in references.items()}
        replica=max(float((actual-torch.load(a.output/f'PRIVATE-rank{i}-loss.pt',weights_only=True)).abs().max()) for i in range(world))
        report=dict(status='COMPLETE_NATIVE_READOUT_PRECISION_DIAGNOSIS',world=world,maximum_native_target_loss_error=error,independent_fp64_agreement=float((references['sdpa64']-references['explicit64']).abs().max()),replica_maximum_error=replica,native_passes_original_threshold_vs_fp64=error['sdpa64']<=.0002,threshold_unchanged=.0002,optimizer_updates=0,scientific_quality_aggregation=False,wall_seconds=time.time()-start)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);main(p.parse_args())
