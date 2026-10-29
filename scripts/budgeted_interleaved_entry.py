"""Qualification only: no scientific training entry is provided."""
import argparse
import os
import runpy
import sys

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry', required=True, choices=['qualification', 'audit'])
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    from compute_trainium.budgeted_interleaved_factory import make_model, matrix_work
    import compute_trainium.interleaved_model_factory as scientific
    import compute_trainium.interleaved_objective as cpu
    import compute_trainium.grouped_interleaved_objective as native
    import compute_trainium.interleaved_workspace as ledger
    scientific.make_model = make_model
    cpu.make_model = make_model
    native.make_model = make_model
    ledger.matrix_work = matrix_work
    runpy.run_module({'qualification': 'scripts.qualify_interleaved_grouped',
                     'audit': 'scripts.audit_interleaved_warm'}[args.entry], run_name='__main__')


if __name__ == '__main__':
    main()
