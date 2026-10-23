"""Actual-wrapper versus separate-projection untied CPU forward/full gradient."""
import argparse,copy,json
from pathlib import Path
import torch
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig,build_preln_recurrent
from compute_trainium.model import ModelConfig
from compute_trainium.independent_untied import IndependentUntiedModel


def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 c=PreNormRecurrenceConfig(dim=32,layers=4,heads=4,kv_heads=2,ffn_dim=64,vocab_size=128,max_seq=64,prelude=1,core=2,coda=1,untied=True)
 torch.manual_seed(43);native=build_preln_recurrent(c)
 base={k:getattr(c,k) for k in ModelConfig.__dataclass_fields__}
 independent=IndependentUntiedModel(ModelConfig(**base),prelude=1,core=2,coda=1);independent.load_titan_state(native.state_dict())
 x=torch.randint(0,128,(2,12));actual=independent(x);expected=native(x)
 torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5);actual.square().mean().backward();expected.square().mean().backward()
 exported=copy.deepcopy(native)
 with torch.no_grad():
  for p,q in zip(exported.parameters(),native.parameters(),strict=True):p.copy_(q.grad)
 oracle=IndependentUntiedModel(ModelConfig(**base),prelude=1,core=2,coda=1);oracle.load_titan_state(exported.state_dict())
 maximum=0.
 for p,q in zip(independent.parameters(),oracle.parameters(),strict=True):
  torch.testing.assert_close(p.grad,q,atol=1e-7,rtol=1e-4);maximum=max(maximum,float((p.grad-q).detach().abs().max()))
 a.output.mkdir(parents=True,exist_ok=True)
 report={'status':'PASS_INDEPENDENT_UNTIED_CPU_REFERENCE','maximum_logit_error':float((actual-expected).detach().abs().max()),'maximum_gradient_error':maximum,'all_parameter_gradients_agree':True,'scope':'CPU numerical fixture, not research examples or optimizer updates'}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
