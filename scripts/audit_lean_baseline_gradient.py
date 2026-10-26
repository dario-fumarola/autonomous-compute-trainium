"""Pinned-upstream full-global gradient for one frozen lean baseline."""
import argparse
import functools
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);a,_=p.parse_known_args()
    probe=json.loads(a.plan.read_text());parent_file=a.root/probe['parent_plan'];parent=json.loads(parent_file.read_text())
    from compute_trainium.lean_baseline_science import make_reference,validate_point
    from scripts import audit_low_depth_gradient as engine
    role=parent['model_variant'];engine.make_reference=lambda c,recurrent:make_reference(c,role)
    engine.validate_probe_state=functools.partial(validate_point,root=a.root,parent_file=parent_file)
    engine.main()


if __name__=='__main__':main()
