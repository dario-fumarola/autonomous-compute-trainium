"""Explicit primitive-attention constructor for a frozen scientific gradient point."""
import argparse
import json
import os
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--plan',type=Path,required=True)
    a,_=p.parse_known_args();plan=json.loads(a.plan.read_text())
    from compute_trainium.future_span_primitive_driver import worker_sources
    worker_sources(plan)
    from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
    from scripts import capture_future_span_gradient as engine
    engine.FutureSpanCandidate=FutureSpanMathSDPACandidate
    engine.main()


if __name__=='__main__':main()
