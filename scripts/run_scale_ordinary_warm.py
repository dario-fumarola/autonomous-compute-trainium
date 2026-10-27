"""Prospective conventional 953M numerical/throughput comparator, not science."""
import argparse
from pathlib import Path

from scripts import run_scale_model_warm as capture


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    capture.IDENTITY='scale-953m-ordinary-numerical-v1'
    capture.PUBLIC=Path('results/scale_953m_ordinary_numerical_20261027')
    capture.run(args.root) if args.run else capture.prepare(args.root,role='ordinary8')
