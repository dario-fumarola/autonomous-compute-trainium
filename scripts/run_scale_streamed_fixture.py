"""Qualify bounded owner workspace under the original all-chip fixture clock."""
import argparse
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from scripts import run_scale_sharding_fixture as fixture

IDENTITY='scale-streamed-sharding-fixture-v1'
PUBLIC=Path('results/scale_streamed_sharding_fixture_20261027')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    fixture.IDENTITY=IDENTITY;fixture.PUBLIC=PUBLIC
    if args.run:
        fixture.run(args.root);return
    prior=args.root/'scale-sharding-fixture-v1'
    record=json.loads((prior/'report.json').read_text())
    if record['status']!='PASS_SCALE_SHARDED_SOFTWARE_FIXTURE_AND_REPLAY':
        raise ValueError('Completed original sharded optimizer fixture required')
    deadline=json.loads((prior/'execution/execution-start.json').read_text())['deadline_unix']
    if time.time()+600>deadline:raise TimeoutError('Both numerical phases plus margin must fit original clock')
    fixture.prepare(args.root)
    file=args.root/IDENTITY/'plan.json';plan=json.loads(file.read_text())
    plan.update(optimizer_execution='streamed',absolute_deadline_unix=deadline,
        seconds_limit=min(600,deadline-time.time()),phase_seconds_limit=300,
        physical_change='Direct CPU-to-local loading, per-matrix owner gathering and broadcast; same complete-matrix FP32 update.')
    write_state(file,plan);write_state(PUBLIC/'plan.json',plan)


if __name__=='__main__':main()
