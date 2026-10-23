"""Full-size initial function equality for the sharing intervention."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import torch
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig,build_preln_recurrent
from scripts.qualify_native import digest


def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(8)
 c=PreNormRecurrenceConfig();torch.manual_seed(23);shared=build_preln_recurrent(c)
 torch.manual_seed(23);untied=build_preln_recurrent(replace(c,untied=True))
 initial=digest(shared.state_dict());expected=json.loads(Path('/work/runs/preln-recurrent-v1-s23-u384/report.json').read_text())['initial_model_digest'] if Path('/work/runs/preln-recurrent-v1-s23-u384/report.json').exists() else '00c98060a3068fbbd1bbf5cfbf0dd6ec09e57dbe6c0f70cf66dd31dd3f763de4'
 assert initial==expected
 for key,value in shared.decoder.state_dict().items():assert torch.equal(value,untied.decoder.state_dict()[key])
 for j in range(c.core):
  for p,q in zip(shared.decoder.layers[str(c.prelude+j)].parameters(),untied.extra_layers[j].parameters(),strict=True):assert torch.equal(p,q)
 x=torch.tensor([[1,23,45,67]])
 with torch.no_grad():assert torch.equal(shared(x),untied(x))
 report={'status':'PASS_FULL_SIZE_UNTIED_INITIALIZATION','shared_initial_digest':initial,
  'full_untied_initial_digest':digest(untied.state_dict()),'base_weights_identical':True,'copied_second_core_weights_identical':True,
  'initial_function_bit_identical':True,'parameters_shared':c.parameter_count(),'parameters_untied':replace(c,untied=True).parameter_count(),
  'scope':'Full-size initial weights and a numerical input fixture; no research examples or training updates'}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
