"""Explicit tied-model entry points; existing frozen engines stay unchanged."""
import argparse,os,runpy,sys

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry',required=True,choices=['qualification','audit','train','scientific-audit','evaluate'])
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    from compute_trainium.tied_interleaved_ordinary import make_model
    import compute_trainium.interleaved_model_factory as scientific
    import compute_trainium.interleaved_objective as cpu
    import compute_trainium.grouped_interleaved_objective as native
    scientific.make_model=make_model;cpu.make_model=make_model;native.make_model=make_model
    module={'qualification':'scripts.qualify_interleaved_grouped','audit':'scripts.audit_interleaved_warm',
            'train':'scripts.train_interleaved_screen','scientific-audit':'scripts.audit_interleaved_screen',
            'evaluate':'scripts.evaluate_interleaved_screen'}[args.entry]
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
