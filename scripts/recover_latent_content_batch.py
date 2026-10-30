"""Execution-only CPU readout recovery. Original scientific payload and deadline stay fixed."""
import argparse
import copy
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import readout_latent_content as readout

ORIGINAL = readout.ID
ID = 'latent-exposure-content-diagnostic-recovery-v1'
PUBLIC = Path('results/latent_exposure_20261030/content-recovery-v1')
MODULE = 'scripts.recover_latent_content_batch'


def science(plan):
    ignored = {'identity','registered_unix','threads','batch_broad','source_sha256','recovery'}
    return {k:v for k,v in plan.items() if k not in ignored}


def validate(original, recovered):
    if original['absolute_deadline_unix'] != recovered['absolute_deadline_unix']:
        raise AssertionError('Deadline extension forbidden')
    if science(original) != science(recovered):
        raise AssertionError('Scientific payload changed')


def prepare(root):
    old = readout.read(root/ORIGINAL/'plan.json')
    receipt = readout.read(root/ORIGINAL/'report.json')
    stop = readout.read(root/ORIGINAL/'operational-stop.json')
    benchmark = readout.read(root/ORIGINAL/'batch-benchmark.json')
    if (receipt['status'] != 'STOPPED_LATENT_CONTENT_DIAGNOSTIC' or
            stop['status'] != 'OPERATIONAL_STOP_BEFORE_ANY_COMPLETE_SCORES' or
            list((root/ORIGINAL).glob('PRIVATE-*.json')) or benchmark['quality_scores_computed'] != 0):
        raise AssertionError('Preserved operational stop without complete quality scores required')
    if benchmark['status'] != 'COMPLETE_NUMERICAL_THROUGHPUT_CHECK':
        raise AssertionError('Complete throughput qualification required')
    rows=benchmark['rows']
    if len(rows)!=3 or any(r['batch_single_max_logp_error']>1e-8 or r['peak_rss_bytes']>old['maximum_worker_rss_bytes'] for r in rows):
        raise AssertionError('Qualified batch numerics/RSS required')
    best=min(rows,key=lambda r:r['seconds_per_view']); now=time.time()
    projected=2*384*best['seconds_per_view']+900
    if projected > old['absolute_deadline_unix']-now:
        raise RuntimeError('Projected full readout does not fit original deadline')
    changed=[f for f,h in old['source_sha256'].items() if file_sha256(f)!=h]
    if changed != ['scripts/readout_latent_content.py']:
        raise AssertionError('Only worker dispatch may change among original sources')
    new=copy.deepcopy(old)
    new.update(identity=ID,registered_unix=now,threads=best['threads'],batch_broad=best['batch'])
    new['source_sha256']={f:file_sha256(f) for f in old['source_sha256']}
    for f in ('scripts/recover_latent_content_batch.py','tests/test_latent_content_batch_recovery.py','docs/LATENT_CONTENT_BATCH_RECOVERY_20261030.md'):
        new['source_sha256'][f]=file_sha256(f)
    new['recovery']={'original_plan_sha256':file_sha256(root/ORIGINAL/'plan.json'),
        'original_report_sha256':file_sha256(root/ORIGINAL/'report.json'),
        'stop_sha256':file_sha256(root/ORIGINAL/'operational-stop.json'),
        'benchmark_sha256':file_sha256(root/ORIGINAL/'batch-benchmark.json'),
        'projected_remaining_seconds':projected,'selected_execution':best,
        'original_wall_seconds':receipt['wall_seconds'],'scientific_payload_unchanged':True}
    validate(old,new); readout.guard(root,new)
    (root/ID).mkdir(); PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json',new); write_state(PUBLIC/'plan.json',new)
    write_state(PUBLIC/'original-stop.json',stop); write_state(PUBLIC/'batch-benchmark.json',benchmark)
    print(json.dumps({'status':'REGISTERED_BATCH_ONLY_READOUT_RECOVERY','batch':best['batch'],'threads':best['threads'],
        'projected_seconds':projected,'deadline_unix':new['absolute_deadline_unix']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=('prepare','run','worker'),required=True);p.add_argument('--mode',choices=readout.MODES);a=p.parse_args()
    if a.action=='prepare': prepare(a.root)
    else:
        readout.ID=ID;readout.PUBLIC=PUBLIC
        if a.action=='worker':readout.worker(a.root,a.mode)
        else:readout.run(a.root,worker_module=MODULE)
