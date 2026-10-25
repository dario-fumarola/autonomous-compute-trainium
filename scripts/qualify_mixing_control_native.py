"""Owned finite qualification using unchanged tested update/recovery machinery."""
import argparse
import json
import os
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan',type=Path,required=True)
    args,_=parser.parse_known_args()
    plan=json.loads(args.plan.read_text())
    if plan.get('model_variant') != 'all_mixes_then_one_core_fullgrad':
        raise ValueError('Explicit cheap-control qualification plan required')
    from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate,mixing_only_ledger
    from scripts import qualify_future_span_native as engine
    engine.FutureSpanCandidate=MixingOnlyFutureSpanCandidate
    engine.matmul_ledger=mixing_only_ledger
    engine.main()


if __name__=='__main__':main()
