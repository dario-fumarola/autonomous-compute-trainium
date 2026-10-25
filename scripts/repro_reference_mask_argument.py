"""Small CPU-only API observation; all token arrays are software fixtures."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    if a.output.exists():raise FileExistsError('Preserve previous reference API receipt')
    if subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned MIT reference required')
    sys.path.insert(0,str(a.upstream));torch.set_num_threads(2);torch.manual_seed(137)
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=128,vocab_size=128)
    model=make_reference(c,True).eval()
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(torch.randn_like(parameter)*.01)
    before=digest(dict(sorted(model.state_dict().items())));noise=initial_state(c,29,1381)[:,:48]
    model.initialize_state=lambda _:noise.clone()
    tokens=torch.arange(48)[None];changed=tokens.clone();changed[:,:24]=(changed[:,:24]+17)%128
    mask=torch.block_diag(torch.ones(24,24,dtype=torch.bool).tril(),torch.ones(24,24,dtype=torch.bool).tril())
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        kwargs={'num_steps_pair':torch.tensor([0,1]),'return_logits':True}
        ordinary=model(tokens,**kwargs)['logits'][:,24:]
        masked=model(tokens,attention_mask=mask,**kwargs)['logits'][:,24:]
        altered=model(changed,attention_mask=mask,**kwargs)['logits'][:,24:]
    no_effect=float((ordinary-masked).abs().max());prefix_effect=float((masked-altered).abs().max())
    if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Reproducer changed weights')
    row={'status':'REPRODUCED_REFERENCE_MASK_ARGUMENT_NOT_APPLIED' if no_effect==0 and prefix_effect>2e-5 else 'OBSERVED_REFERENCE_MASK_API_BEHAVIOR',
        'upstream_commit':PIN,'mask_vs_unmasked_max_abs':no_effect,'removed_prefix_change_with_mask_max_abs':prefix_effect,
        'source_sha256':file_sha256(__file__),'assigned_chips':0,'optimizer_updates':0,'research_examples':0,
        'wall_seconds':time.time()-started,'scope':'Tiny random perturbed CPU model,48 numerical token-index fixtures,no prediction targets,NLL or research corpus. Optional public attention_mask argument accepted but not applied on this pinned mixer path. Installed/upstream files remain unchanged; no upstream submission.'}
    write_state(a.output,row);print(json.dumps(row),flush=True)


if __name__=='__main__':main()
