"""Explicit scientific entry for a separately registered supported-budget fit."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry', required=True, choices=['train', 'scientific-audit', 'evaluate'])
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
    runpy.run_module({'train': 'scripts.train_interleaved_screen',
                     'scientific-audit': 'scripts.audit_interleaved_screen',
                     'evaluate': 'scripts.evaluate_interleaved_screen'}[args.entry], run_name='__main__')
    if args.entry == 'train' and int(os.environ.get('RANK', '0')) == 0:
        # Preserve the qualified engine's numbers; correct its legacy static
        # description without changing that frozen training implementation.
        output = Path(remaining[remaining.index('--output')+1])
        receipt = json.loads((output/'report.json').read_text())
        receipt['legacy_engine_scope'] = receipt['scope']
        receipt['scope'] = 'Explicitly trained zero/one/two generated-state budgets. Telemetry extra_steps contains graph IDs, not token counts. Uniform real-token NTP; independent exact sequential readout remains required.'
        from compute_trainium.budgeted_interleaved_factory import GRAPH_SPECS
        receipt['graph_specs'] = {str(k): dict(thought_budget=b, numerical_refinements=r) for k, (b, r) in GRAPH_SPECS.items()}
        receipt['warm_timing_scope'] = 'Legacy engine mean excludes first three updates only; includes first use of graphs24/25. Full fit wall time retains all first-use and audit costs.'
        (output/'report.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
