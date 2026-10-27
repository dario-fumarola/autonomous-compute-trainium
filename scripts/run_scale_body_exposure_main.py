"""The frozen8B/10.835B corrected-scale point, never automatic confirmation."""
import argparse
from pathlib import Path
from scripts import run_scale_main as engine


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    engine.IDENTITY='scale-body-exposure-main-v1'
    engine.PUBLIC=Path('results/scale_body_exposure_main_20261027')
    engine.run(a.root) if a.run else engine.prepare(a.root,boundary_corrected=True,body_exposure=True)
