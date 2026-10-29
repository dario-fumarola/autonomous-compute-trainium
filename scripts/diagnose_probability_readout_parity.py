"""Zero-update precision diagnosis on the exact batch rejected by readout parity."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import torch
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.probability_intervention import ProposalReadout
from compute_trainium.training import file_sha256


def main(root,out):
    start=time.time();out.mkdir(exist_ok=False);torch.set_num_threads(12)
    p=json.loads((root/'probability-workspace-screen-recovery-v1/plan.json').read_text())
    phase=p['arms']['workspace']['phase'];checkpoint=root/'runs'/phase/'PRIVATE-u1530.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    xs=np.load(root/p['evaluation_relative']/'broad-tokens.npy',mmap_mode='r')[128:132]
    x=torch.from_numpy(np.asarray(xs[:,:-1],dtype=np.int64));y=torch.from_numpy(np.asarray(xs[:,1:],dtype=np.int64))
    perm=torch.randperm(p['model_config']['vocab_size'],generator=torch.Generator().manual_seed(p['permutation_seed']))
    receipt=dict(purpose='Readout precision diagnosis only,no quality aggregation or model selection',checkpoint_sha256=file_sha256(checkpoint),source_sha256=file_sha256(__file__),broad_indices=[128,129,130,131],extra_steps=2,permutation_seed=p['permutation_seed'],optimizer_updates=0,assigned_chips=0,registered_unix=start)
    (out/'plan.json').write_text(json.dumps(receipt,indent=2)+'\n');values={}
    for name,explicit,dtype in [('sdpa32',False,torch.float32),('explicit32',True,torch.float32),('sdpa64',False,torch.float64),('explicit64',True,torch.float64)]:
        m=ProbabilityWorkspace(ModelConfig(**p['model_config']),explicit_attention=explicit).to(dtype);m.load_state_dict(saved['model'])
        with torch.no_grad():
            logits=ProposalReadout(m,2,permutation=perm)(x)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
        values[name]=loss.double();del m,logits
    differences={f'{a}_vs_{b}':float((values[a]-values[b]).abs().max()) for a,b in [('sdpa32','explicit32'),('sdpa32','sdpa64'),('explicit32','sdpa64'),('sdpa64','explicit64')]}
    result=dict(status='COMPLETE_READOUT_PRECISION_DIAGNOSIS',maximum_target_loss_differences=differences,quality_scores_published=False,optimizer_updates=0,assigned_chips=0,wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='Numerical comparison on the frozen failed check batch only;not scientific quality or an SDK defect attribution.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
