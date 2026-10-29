"""Reuse frozen single-family LR selection with truthful untied metadata."""
import argparse,contextlib,io,json
from scripts.analyze_interleaved_tied_tuning import main as select


def main(a):
    with contextlib.redirect_stdout(io.StringIO()):select(a)
    path=a.output/'report.json';result=json.loads(path.read_text())
    result.update(status='COMPLETE_ORDINARY_FRONTIER_TUNING_ANALYSIS',
        scope='Single-seed pre-inventoried untied ordinary-shape LR selection on reused DEV. No unbiased confirmation or exhaustive conventional frontier claim. Latent comparison uses exact sequential predictions.')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
