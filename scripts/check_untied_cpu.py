"""Actual-wrapper function/gradient equality when shared core copies untie."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import torch
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig,build_preln_recurrent
from scripts.qualify_native import digest


def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 c=PreNormRecurrenceConfig(dim=32,layers=4,heads=4,kv_heads=2,ffn_dim=64,vocab_size=128,max_seq=64,prelude=1,core=2,coda=1,repetitions=2)
 torch.manual_seed(23);shared=build_preln_recurrent(c)
 torch.manual_seed(23);untied=build_preln_recurrent(replace(c,untied=True))
 x=torch.randint(0,128,(2,12));s=shared(x);u=untied(x)
 torch.testing.assert_close(s,u,rtol=0,atol=0);s.square().mean().backward();u.square().mean().backward()
 other=dict(untied.named_parameters())
 for name,p in shared.named_parameters():
  expected=other[name].grad
  for layer in range(c.prelude,c.prelude+c.core):
   prefix=f'decoder.layers.{layer}.'
   if name.startswith(prefix):expected=expected+other[f'extra_layers.{layer-c.prelude}.'+name[len(prefix):]].grad
  torch.testing.assert_close(p.grad,expected,atol=1e-7,rtol=1e-4)
 changed=x.clone();changed[:,7:]=(changed[:,7:]+4)%128
 torch.testing.assert_close(untied(x)[:,:7],untied(changed)[:,:7],atol=1e-6,rtol=1e-5)
 assert all(p.grad is not None and p.grad.norm()>0 for p in untied.parameters())
 report={'status':'PASS_ACTUAL_UNTIED_FUNCTION_GRADIENT_CPU','initial_function_bit_identical':True,
  'shared_core_gradients_equal_sum_of_independent_copies':True,'all_untied_parameters_receive_gradients':True,
  'causal_future_invariance':True,'shared_parameters':c.parameter_count(),'untied_parameters':replace(c,untied=True).parameter_count(),
  'scope':'CPU numerical fixture on actual native-wrapper implementation; no research corpus or model update'}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
