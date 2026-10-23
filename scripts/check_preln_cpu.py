"""Exact CPU equivalence to the preserved ordinary control, plus causal BPTT."""
import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch

from compute_trainium.model import ModelConfig
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig, build_preln_recurrent
from compute_trainium.titan import build_titan
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    c=PreNormRecurrenceConfig(dim=32,layers=4,heads=4,kv_heads=2,ffn_dim=64,
                            vocab_size=128,max_seq=64,prelude=1,core=2,coda=1,repetitions=1)
    base={k:getattr(c,k) for k in ModelConfig.__dataclass_fields__}
    torch.manual_seed(23);ordinary=build_titan(ModelConfig(**base))
    torch.manual_seed(23);one=build_preln_recurrent(c)
    torch.manual_seed(23);two=build_preln_recurrent(replace(c,repetitions=2))
    assert digest(ordinary.state_dict())==digest(one.state_dict())==digest(two.state_dict())
    x=torch.randint(0,128,(2,12));changed=x.clone();changed[:,7:]=(changed[:,7:]+4)%128
    a,b=ordinary(x),one(x)
    assert torch.equal(a,b)
    a.square().mean().backward();b.square().mean().backward()
    for p,q in zip(ordinary.parameters(),one.parameters(),strict=True):assert torch.equal(p.grad,q.grad)
    y=two(x);z=two(changed)
    torch.testing.assert_close(y[:,:7],z[:,:7],atol=1e-6,rtol=1e-5)
    assert (y[:,7:]-z[:,7:]).abs().max()>.001
    y.square().mean().backward()
    assert all(p.grad is not None and p.grad.norm()>0 for p in two.parameters())
    # Full-size initial state is the same as the existing three control runs.
    torch.manual_seed(23);large=build_preln_recurrent(PreNormRecurrenceConfig())
    initial=digest(large.state_dict())
    expected=json.loads(Path('/work/runs/natural-curve-v1-ordinary-s23-u384/report.json').read_text())['initial_model_digest']
    assert initial==expected
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS_PRELN_EXACT_ORDINARY_CPU','ordinary_logits_bit_identical':True,
            'ordinary_full_gradients_bit_identical':True,'one_two_pass_initial_weights_identical':True,
            'causal_future_invariance':True,'all_parameters_receive_gradients':True,
            'full_size_initial_digest_matches_preserved_control':initial,'parameters':large.config.parameter_count()}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
