"""Add the audited longer ordinary endpoint to the unchanged elementary panel.

One CPU-only readout, five-minute cap, four threads,6GiB. Freeze every input
before scoring. Reuse unchanged audited latent/pause scores by their hashes.
This is a descriptive extension, not a new gate or a fresh confirmation panel.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import readout_direct_arithmetic as direct
from scripts.readout_interleaved_bracket_type import checkpoint
from scripts.readout_primitive_arithmetic import statistics

ID='matched-work-primitive-readout-v1'
PUBLIC=Path('results/primitive_arithmetic_20261030/matched-work-v1')


def main(root):
    read=lambda f:json.loads(Path(f).read_text())
    start=time.time();prior=root/'primitive-arithmetic-readout-v1';p=copy.deepcopy(read(prior/'plan.json'))
    if read(prior/'report.json')['status']!='COMPLETE_PRIMITIVE_ARITHMETIC_READOUT':raise AssertionError('Original complete panel readout required')
    point=checkpoint(root,'latent-exposure-work-recovery-v1','ordinary');fit=read(root/point['plan_relative'])
    if fit['seed']!=730271 or fit['arms']['ordinary']['real_targets']!=3782650274:raise AssertionError('Exact original longer control required')
    point.update(seed=730271,role='ordinary')
    retained={r:file_sha256(prior/f'PRIVATE-{r}-scores.json') for r in ('latent','pause')}
    p.update(identity=ID,registered_unix=start,absolute_deadline_unix=start+300,threads=4,points={'ordinary':point},
        retained_score_sha256=retained,source_sha256={f:file_sha256(f) for f in p['source_sha256']},
        scope='Exploratory unchanged elementary panel at matched nominal training matrix work: latent1B versus ordinary3.783B. Trained pauses remain an equal-data attribution control. Basic-fact TRAIN overlap is allowed; no novel-expression, reliable-reasoning or confirmation claim. No gate change or automatic fit.')
    p['source_sha256']['scripts/readout_work_primitive.py']=file_sha256(__file__)
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every attempt')
    direct.guard(root,p);out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    result=dict(status='STOPPED_MATCHED_WORK_PRIMITIVE_READOUT')
    try:
        direct.ID=ID;direct.PANEL='primitive-arithmetic-panel-v1';direct.worker(root,'ordinary')
        receipt=read(out/'ordinary-report.json')
        if receipt['status']!='COMPLETE_DIRECT_ARITHMETIC_WORKER' or len(receipt['checks'])!=3:raise AssertionError('Complete numerical readout required')
        if any(file_sha256(prior/f'PRIVATE-{r}-scores.json')!=h for r,h in retained.items()):raise AssertionError('Cached audited scores changed')
        rows=read(out/'PRIVATE-ordinary-scores.json')+[v for r in retained for v in read(prior/f'PRIVATE-{r}-scores.json')]
        analysis=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']);analysis['scope']=p['scope']
        result.update(status='COMPLETE_MATCHED_WORK_PRIMITIVE_READOUT',statistics=analysis,worker=receipt,
            ordinary_targets=3782650274,latent_targets=1000060368,pause_targets=1000060368)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,
            terminal_alpha_spent=0,automatic_promotion=False,plan_sha256=file_sha256(out/'plan.json'),scope=p['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
