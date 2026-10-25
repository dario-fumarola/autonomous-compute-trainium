"""Measure forward matmuls on actual-size pinned CPU reference operators."""
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
from torch.utils.flop_counter import FlopCounterMode

from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    root=a.root/'parcae-low-depth-pilot-v1';output=Path('results/parcae_low_depth_pilot_20261024/inference-work.json')
    if output.exists():raise FileExistsError('Preserve inference operator-count receipt')
    plan=json.loads((root/'preparation-plan.json').read_text());c=ParcaeCandidateConfig(**plan['model'])
    upstream=a.root/'upstream-parcae-69284c1'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned upstream required')
    sys.path.insert(0,str(upstream));torch.set_num_threads(12);records=[]
    for recurrent in [False,True]:
        key='s1231-'+('family' if recurrent else 'ordinary');initial=plan['initial_states'][key]
        if file_sha256(root/initial['file'])!=initial['sha256']:raise AssertionError('Fresh untrained initialization changed')
        state=torch.load(root/initial['file'],map_location='cpu',mmap=True,weights_only=False)
        model=make_reference(c,recurrent).eval();model.load_state_dict(state['model'],strict=True);del state
        before=digest(dict(sorted(model.state_dict().items())));noise=initial_state(c,1_000_000_007,1381)
        current=None
        if recurrent:model.initialize_state=lambda _:current
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            for positions in [128,1024]:
                tokens=torch.zeros((1,positions),dtype=torch.long) # Numerical operation-count fixture,not research text.
                current=noise[:,:positions].clone()
                for depth in ([1,2,3] if recurrent else [1]):
                    kwargs={'num_steps_pair':torch.tensor([depth-1,1])} if recurrent else {}
                    begin=time.time()
                    with FlopCounterMode(display=False) as counter:logits=model(tokens,return_logits=True,**kwargs)['logits']
                    expected=matmul_ledger(c,positions,recurrent=recurrent,total_steps=depth)['forward_matmul_flops']
                    actual=counter.get_total_flops()
                    if actual!=expected or logits.shape!=(1,positions,32000):raise AssertionError('Independent forward operator count differs from ledger')
                    records.append({'arm':'family' if recurrent else 'ordinary','positions':positions,'depth':depth,
                        'parameters':sum(p.numel() for p in model.parameters()),'counted_forward_matmuls':actual,
                        'ledger_forward_matmuls':expected,'head_forward_matmuls':2*positions*c.dim*c.vocab_size,
                        'cpu_fixture_seconds':time.time()-begin})
                    del logits
        if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Operation counting changed model')
        del model
    rows={r['depth']:r for r in records if r['arm']=='family' and r['positions']==1024}
    result={'status':'PASS_INDEPENDENT_ACTUAL_SIZE_FORWARD_MATMUL_COUNTS','records':records,
        'family_depth2_over_depth1':rows[2]['counted_forward_matmuls']/rows[1]['counted_forward_matmuls'],
        'family_depth3_over_depth1':rows[3]['counted_forward_matmuls']/rows[1]['counted_forward_matmuls'],
        'upstream_commit':PIN,'source_sha256':file_sha256(__file__),'assigned_chips':0,'optimizer_updates':0,
        'research_examples':0,'test_accessed':False,'wall_seconds':time.time()-started,
        'scope':'Independent pinned upstream actual140.8M/142.0M CPU forward matrix-operation counts,FMA=2,full-square attention and full tied vocabulary head included. Numerical token-index fixtures,no target labels or model-quality metrics. Excludes gathers,elementwise work,hardware tile padding,communication and caching. CPU fixture times include instrumentation;not steady native latency or autoregressive deployment cost.'}
    write_state(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
