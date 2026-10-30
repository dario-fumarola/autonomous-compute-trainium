"""Recorded norm-order repair; preserve the first failed qualification."""
import argparse
from pathlib import Path
from compute_trainium.training import file_sha256
from scripts.qualify_bucketed_mature_cpu import read,run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    original=a.root/'bucketed-mature-cpu-v1/report.json';r=read(original)
    if (r['status']!='STOPPED_BUCKETED_MATURE_CPU'
        or r.get('error')!='Registered complete-gradient/update/moment parity limit'):
        raise AssertionError('Preserved initiating precision failure required')
    run(a.root,identity='bucketed-mature-cpu-v2',public=Path('results/gradient_bucketing_20261030/mature-cpu-v2'),
        scope_document='docs/BUCKETED_NORM_REPAIR_20261030.md',
        extra_sources=('scripts/recheck_bucketed_mature_cpu.py',),
        repair_of=dict(identity='bucketed-mature-cpu-v1',report_sha256=file_sha256(original),
                       original_source_commit='106f58d'))
