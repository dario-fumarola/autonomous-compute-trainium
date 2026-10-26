"""Qualified model/optimizer constructors around the existing NTP science loop."""
import argparse
import functools
import json
import os
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--plan',type=Path,required=True)
    a,_=p.parse_known_args();plan=json.loads(a.plan.read_text())
    from compute_trainium.lean_baseline_science import model_factory,signature,worker_sources,update_work,ROLES
    from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
    role=plan['model_variant']
    if role not in ROLES:raise ValueError('Exact prepared baseline role required')
    from scripts import train_low_depth_scientific as engine
    engine.ParcaeFP16Candidate=model_factory(role);engine.ParcaePreciseOptimizer=ParcaeRuntimeScalarOptimizer
    engine.signature=signature;engine.worker_sources=worker_sources;engine.update_work=functools.partial(update_work,role)
    engine.main()


if __name__=='__main__':main()
