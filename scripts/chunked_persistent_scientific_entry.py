"""Explicit scientific dispatch for a separately qualified and frozen screen."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry',required=True,choices=['train','scientific-audit','evaluate'])
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    from compute_trainium.chunked_persistent_factory import make_model,matrix_work,GRAPH_NAMES
    import compute_trainium.interleaved_model_factory as scientific
    import compute_trainium.interleaved_objective as cpu
    import compute_trainium.grouped_interleaved_objective as native
    import compute_trainium.interleaved_workspace as ledger
    scientific.make_model=make_model;cpu.make_model=make_model;native.make_model=make_model
    ledger.matrix_work=matrix_work
    runpy.run_module({'train':'scripts.train_interleaved_screen',
        'scientific-audit':'scripts.audit_interleaved_screen','evaluate':'scripts.evaluate_interleaved_screen'}[args.entry],run_name='__main__')
    if args.entry=='train' and int(os.environ.get('RANK','0'))==0:
        path=Path(remaining[remaining.index('--output')+1])/'report.json'
        report=json.loads(path.read_text());report['graph_names']=GRAPH_NAMES
        report['legacy_engine_scope']=report['scope']
        report['scope']='Sparse persistent workspace, one slot per128 source positions. Legacy extra_steps identifies execution graph, not a thought budget. Uniform genuine-token NTP, exact chronological differentiable KV, no solver or vocabulary sampling. Scientific endpoint requires mature independent audit.'
        path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
