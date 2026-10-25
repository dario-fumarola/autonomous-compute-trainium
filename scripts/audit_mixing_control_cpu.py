"""Independent upstream control;unchanged complete-objective/update audit."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan',type=Path,required=True)
    args,_=parser.parse_known_args()
    if json.loads(args.plan.read_text()).get('model_variant') != 'all_mixes_then_one_core_fullgrad':
        raise ValueError('Explicit cheap-control reference plan required')
    from compute_trainium.mixing_control_reference import construct_reference
    from compute_trainium.parcae_mixing_control import mixing_only_ledger
    from scripts import audit_future_span_qualification as engine
    engine.construct_reference=construct_reference
    engine.matmul_ledger=mixing_only_ledger
    engine.audit()


if __name__=='__main__':main()
