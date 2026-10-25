"""Explicit constructor injection into the unchanged zero-update probe engine.

Only this new process's model-constructor binding changes. The installed SDK,
old source files,checkpoints,optimizer and source cursor remain untouched.
Both the original engine and this entry are in the new probe's exact manifest.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--plan',type=Path,required=True)
    args,_=p.parse_known_args();plan=json.loads(args.plan.read_text())
    if plan.get('execution_variant')!='fp16_body_explicit_fp32_sdpa':raise ValueError('Explicit precision-workaround probe required')
    from compute_trainium.parcae_math_sdpa_candidate import ParcaeMathSDPACandidate
    from scripts import capture_low_depth_gradient as engine
    engine.ParcaeFP16Candidate=ParcaeMathSDPACandidate
    engine.main()


if __name__=='__main__':main()
