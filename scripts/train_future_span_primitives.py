"""Explicit scientific-driver constructor and snapshot policy;no launch policy."""
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
    from scripts import train_future_span_scientific as engine
    engine.FutureSpanCandidate=FutureSpanMathSDPACandidate
    engine.worker_sources=worker_sources
    engine.main()


if __name__=='__main__':main()
