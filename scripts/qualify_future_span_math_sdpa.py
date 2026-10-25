"""Explicit alternate constructor for qualification;installed SDK is untouched."""
import argparse
import json
import os
from pathlib import Path

# Select this rank before importing PyTorch/backend autoload.
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--plan',type=Path,required=True)
    a,_=p.parse_known_args();plan=json.loads(a.plan.read_text())
    if plan.get('execution_variant')!='fp16_body_fp32_attention_primitives':
        raise ValueError('Explicit independently qualified attention path required')
    from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
    from scripts import qualify_future_span_native as engine
    engine.FutureSpanCandidate=FutureSpanMathSDPACandidate
    engine.main()


if __name__=='__main__':main()
