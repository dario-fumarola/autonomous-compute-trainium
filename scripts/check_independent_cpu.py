"""Qualify separate CPU forward and adjoint against actual TorchTitan blocks."""
import argparse
import copy
import json
from pathlib import Path

import torch

from compute_trainium.independent import IndependentModel
from compute_trainium.model import ModelConfig
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig,build_preln_recurrent


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);results=[]
    for repetitions in [1,2,4]:
        c=PreNormRecurrenceConfig(dim=32,layers=4,heads=4,kv_heads=2,ffn_dim=64,
                                vocab_size=128,max_seq=64,prelude=1,core=2,coda=1,repetitions=repetitions)
        torch.manual_seed(19);native=build_preln_recurrent(c)
        reference=IndependentModel(ModelConfig(**{k:getattr(c,k) for k in ModelConfig.__dataclass_fields__}),
                                   repetitions=repetitions,prelude=1,core=2,coda=1)
        reference.load_titan_state(native.state_dict())
        x=torch.randint(0,128,(2,13));a=native(x);b=reference(x)
        torch.testing.assert_close(a,b,atol=5e-6,rtol=2e-5)
        a.square().mean().backward();b.square().mean().backward()
        # Native parameters store fused QKV; its state_dict deliberately exports
        # separate projections. Export gradients through that same documented
        # layout conversion, rather than assuming parameter/state key identity.
        gradient_source=copy.deepcopy(native)
        with torch.no_grad():
            for target,param in zip(gradient_source.parameters(),native.parameters(),strict=True):
                target.copy_(param.grad)
        mapped=gradient_source.state_dict()
        gradient=IndependentModel(reference.config,repetitions,1,2,1)
        gradient.load_titan_state(mapped)
        for expected,actual in zip(gradient.parameters(),reference.parameters(),strict=True):
            torch.testing.assert_close(expected,actual.grad,atol=3e-6,rtol=2e-4)
        results.append({'repetitions':repetitions,'maximum_logit_error':float((a-b).detach().abs().max()),'full_gradients_agree':True})
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS_INDEPENDENT_CPU_FORWARD_AND_FULL_GRADIENT','results':results}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
