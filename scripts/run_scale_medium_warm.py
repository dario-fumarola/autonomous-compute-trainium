"""Measured-budget medium scale qualification; no scientific fit or LR choice."""
import argparse
from pathlib import Path

from scripts import run_scale_model_warm as capture


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--role',choices=['fixed_two','ordinary8'],required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    capture.IDENTITY='scale-376m-fixed-two-numerical-v1' if args.role=='fixed_two' else 'scale-375m-ordinary-numerical-v1'
    capture.PUBLIC=Path('results/scale_medium_numerical_20261027')/args.role
    capture.run(args.root) if args.run else capture.prepare(args.root,role=args.role,profile='medium')
