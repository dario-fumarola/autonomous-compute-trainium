"""Apply the unchanged registered consistency diagnostic to all mixed endpoints."""
import argparse
import copy
import json
import time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts import diagnose_interleaved_convergence as diagnostic

ID='interleaved-mixed-convergence-v1'
TUNING='interleaved-mixed-round-tuning-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/mixed-convergence-v1')


def prepare(root):
    prior_path=root/'interleaved-convergence-v1/plan.json'
    old=json.loads(prior_path.read_text())
    p=copy.deepcopy(old)
    training=root/TUNING/'plan.json'
    training_plan=json.loads(training.read_text())
    if set(training_plan['arms'])!={'mixed-lr0','mixed-lr1','mixed-lr2'}:
        raise AssertionError('All three registered mixed endpoints required')
    if any(file_sha256(f)!=h for f,h in old['source_sha256'].items()):
        raise AssertionError('Original diagnostic computation must remain unchanged')
    p['source_sha256']['scripts/diagnose_interleaved_mixed_convergence.py']=file_sha256(__file__)
    p['source_sha256']['docs/INTERLEAVED_MIXED_ROUND_TUNING_DRAFT_20261029.md']=file_sha256('docs/INTERLEAVED_MIXED_ROUND_TUNING_DRAFT_20261029.md')
    now=time.time()
    p.update(identity=ID,registered_unix=now,absolute_deadline_unix=now+5400,
        tuning_plan_sha256=file_sha256(training),prior_diagnostic_plan_sha256=file_sha256(prior_path),
        arms=['mixed-lr0','mixed-lr1','mixed-lr2'])
    (root/ID).mkdir();PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    diagnostic.ID=ID;diagnostic.TUNING=TUNING;diagnostic.PUBLIC=PUBLIC
    diagnostic.run(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else run)(a.root)
