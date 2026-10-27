"""Conditional frozen boundary-corrected development comparison,never confirmation."""
import argparse
from pathlib import Path
from scripts import run_scale_main as engine

IDENTITY='scale-boundary-main-v1'
PUBLIC=Path('results/scale_boundary_main_20261027')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC
    engine.run(a.root) if a.run else engine.prepare(a.root,boundary_corrected=True)
