"""Explicit lean constructor for the existing qualification-only trainer."""
import argparse
import json
import os
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--preparation-root',type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--accumulation',type=int,required=True)
    p.add_argument('--precision',required=True);a,_=p.parse_known_args()
    from compute_trainium.training import file_sha256
    plan=json.loads((a.preparation_root/'plan.json').read_text())
    if (plan.get('execution_variant')!='lean_fp16_primitive_attention_runtime_scalars'
            or a.arm not in ['fixed1','poisson2'] or a.accumulation!=16 or a.precision!='fp16'):
        raise ValueError('Only the separately frozen two lean qualification cases are eligible')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/file)!=sha for file,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Frozen lean worker source changed')
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
    from scripts import train_parcae_precise_qualification as engine
    engine.ParcaeFP16Candidate=ParcaePrimitiveSDPACandidate
    engine.ParcaeOptimizerCandidate=ParcaeRuntimeScalarOptimizer
    engine.configure_optimizer=changing_qualification_schedule
    engine.main()


if __name__=='__main__':main()
