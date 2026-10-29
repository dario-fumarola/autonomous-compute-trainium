"""Disposable qualification dispatch; no scientific fit entry."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


if __name__ == '__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry',required=True,choices=['qualification','audit'])
    args,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    from compute_trainium.chunked_persistent_factory import make_model,matrix_work,GRAPH_NAMES
    import compute_trainium.interleaved_objective as cpu
    import compute_trainium.grouped_interleaved_objective as native
    import compute_trainium.interleaved_workspace as ledger
    cpu.make_model=make_model
    native.make_model=make_model
    ledger.matrix_work=matrix_work
    runpy.run_module({'qualification':'scripts.qualify_interleaved_grouped','audit':'scripts.audit_interleaved_warm'}[args.entry],run_name='__main__')
    if args.entry=='qualification' and int(os.environ.get('RANK','0'))==0:
        path=Path(remaining[remaining.index('--output')+1])/'report.json'
        report=json.loads(path.read_text())
        report['graph_names']=GRAPH_NAMES
        report['scope']='Disposable sparse persistent qualification. Legacy extra_steps/warm_by_depth fields identify full ordinary, chunked ordinary, generated and constant graphs, not latent budgets. All workspace graphs insert exactly one slot per128 source tokens. No scientific weight reuse or held-out scoring.'
        path.write_text(json.dumps(report,indent=2)+'\n')
