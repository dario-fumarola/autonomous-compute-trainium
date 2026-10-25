"""Same qualified full-model numerical path;compare optimizer execution only."""
import argparse
import json
import os
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args();plan=json.loads(args.plan.read_text())
    variant=plan['optimizer_variants'][args.case]
    if variant not in ['replicated','owner'] or not plan['qualification_only']:
        raise ValueError('Explicit non-adopted execution screen required')
    from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
    from compute_trainium.owner_broadcast_muon_candidate import OwnerBroadcastOptimizerCandidate
    from scripts import qualify_future_span_native as engine
    engine.FutureSpanCandidate=FutureSpanMathSDPACandidate
    if variant=='owner':engine.ParcaeRuntimeScalarOptimizer=OwnerBroadcastOptimizerCandidate
    engine.main()
    if int(os.environ['RANK'])==0:
        p=args.output/'report.json';r=json.loads(p.read_text())
        r.update(optimizer_physical_execution=variant,full_training_qualified=False,scientific_adoption=False)
        if variant=='owner':
            for key in ['optimizer_matmuls_all_replicas','warm_optimizer_matmuls_all_replicas']:
                r['replicated_recipe_'+key]=r[key]
                if r[key]%32:raise AssertionError('Exact one-owner work geometry required')
                r[key]//=32
        r['scope']='Full one-pass training throughput screen on existing natural TRAIN qualification windows. Same original body,head,gradients,schedule and global batch;only optimizer execution differs. All reported updates are non-adopted qualification. Native full-state comparison is diagnostic;independent CPU updates,recovery and hardware-swapped timing are still required before adoption.'
        p.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
