"""Original complete NTP capture with the explicitly qualified baseline model."""
import argparse
import functools
import json
import os
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);a,_=p.parse_known_args()
    probe=json.loads(a.plan.read_text());parent_file=a.root/probe['parent_plan'];parent=json.loads(parent_file.read_text())
    from compute_trainium.lean_baseline_science import model_factory,validate_point
    from scripts import capture_low_depth_gradient as engine
    role=parent['model_variant'];engine.ParcaeFP16Candidate=model_factory(role)
    engine.validate_probe_state=functools.partial(validate_point,root=a.root,parent_file=parent_file)
    if role=='mixing':
        from compute_trainium.parcae_mixing_control import mixing_only_ledger
        engine.matmul_ledger=lambda c,context,batch=1,**kw:mixing_only_ledger(c,context,batch,
            total_steps=kw.get('total_steps',1),backward_cap=kw.get('backward_cap',1))
    engine.main()


if __name__=='__main__':main()
