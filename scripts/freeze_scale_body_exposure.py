"""Freeze a data/model-balanced budget before any joint-scale main outcome."""
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='scale-body-exposure-scope-v1'
PUBLIC=Path('results/scale_body_exposure_scope_20261027')
DOCUMENT='docs/SCALE_BODY_EXPOSURE_DEVELOPMENT_20261027.md'


def first_full_update_at_least(counts,target,global_windows=512):
    lo,hi=0,(len(counts)-1)//global_windows
    def total(u):return sum(map(int,counts[u*global_windows]))
    if total(hi)<target:raise ValueError('Accepted capacity cannot cover endpoint')
    while lo<hi:
        mid=(lo+hi)//2
        if total(mid)<target:lo=mid+1
        else:hi=mid
    return lo


def freeze(root):
    import numpy as np
    import torch
    from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
    from compute_trainium.scale_pilot_size import profiled_inventory
    from compute_trainium.scale_work_ledger import fixed_update_work,work_matched_updates
    if any((root/name).exists() for name in ['scale-medium-main-v1','scale-boundary-main-v1','scale-body-exposure-main-v1']):
        raise ValueError('No main may be admitted or observed before this new scope')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every prospective scope')
    clock=root/'research-clock-accounting-amendment-20261027-v1/plan.json'
    if not json.loads(clock.read_text())['overall_clock_admission_waived']:
        raise ValueError('Explicit owner clock direction required')
    initial=root/'scale-main-original-initializers-v1';initial_report=json.loads((initial/'report.json').read_text())
    if (initial_report['seed']!=49177 or initial_report['optimizer_updates']!=0
            or initial_report['model_forwards']!=0):raise ValueError('Original unused main C0 required')
    for row in initial_report['records']:
        if file_sha256(initial/row['file'])!=row['sha256']:raise AssertionError('Original C0 bytes changed')
    old=Path('results/fixed_two_small_core_main_20261025/plan.json')
    legacy=ParcaeCandidateConfig(**json.loads(old.read_text())['model'])
    inventory=profiled_inventory('medium');bodies=[]
    for label,cfg,recurrent in [('legacy_142M',legacy,True)]+[
            (r['role'],ParcaeCandidateConfig(**r['config']),r['role']=='fixed_two') for r in inventory]:
        with torch.device('meta'):model=ParcaeCandidate(cfg,recurrent=recurrent,initialize=False)
        lookup={id(m.weight):m.weight.numel() for m in model.modules() if isinstance(m,torch.nn.Embedding)}
        total=sum(p.numel() for p in model.parameters())
        if total!=cfg.parameter_count(recurrent):raise AssertionError('Independent actual/meta inventory differs')
        bodies.append(dict(role=label,parameters=total,lookup_parameters=sum(lookup.values()),
                           non_lookup_parameters=total-sum(lookup.values())))
    data=root/'natural-append-capacity-v4';manifest=data/'packed/manifest.json'
    counts=np.load(data/'packed/prefix_counts.npy',mmap_mode='r')
    def targets(u):return sum(map(int,counts[u*512]))
    r_end=first_full_update_at_least(counts,8_000_000_000)
    rw=fixed_update_work(inventory[0]['config'],'fixed_two')['total_per_update']
    ow=fixed_update_work(inventory[1]['config'],'ordinary8')['total_per_update']
    match=work_matched_updates(rw,r_end,ow);o_end=match['updates']
    if (r_end,targets(r_end),o_end,targets(o_end))!=(15286,8000125097,20703,10835106429):
        raise AssertionError('Prospective declared source/work endpoint differs')
    qualification=[];seconds=[]
    for name in ['scale-boundary-fixed-two-numerical-v1','scale-65536-ordinary-numerical-v1']:
        proof=root/name/'completion-verification.json';report=root/name/'report.json'
        status=json.loads(proof.read_text())['status']
        if not status.startswith('PASS_') or 'COMPLETE_NUMERICAL_QUALIFICATION' not in status:
            raise ValueError('Both complete exact numerical proofs required')
        warm=json.loads(report.read_text())['records']['warm']
        if warm['warm_updates']!=100:raise ValueError('Actual100-update measurement required')
        seconds.append(warm['warm_seconds']/100)
        qualification.append(dict(identity=name,proof_sha256=file_sha256(proof),report_sha256=file_sha256(report)))
    r_stable=int(r_end*.9);o_stable=int(o_end*.9)
    warm=(r_end*seconds[0]+(o_end+r_end-r_stable)*seconds[1])
    value=dict(identity=IDENTITY,registered_unix=time.time(),scope_sha256=file_sha256(DOCUMENT),scope=Path(DOCUMENT).read_text(),
        source_sha256=file_sha256(Path(__file__)),original_unused_initializer_report_sha256=file_sha256(initial/'report.json'),
        original_unadmitted_boundary_scope_sha256=file_sha256(root/'scale-boundary-scope-freeze-v1/scope.json'),
        old_precision_scope_document_sha256=json.loads((root/'scale-boundary-scope-freeze-v1/scope.json').read_text())['scope_sha256'],
        user_clock_amendment_sha256=file_sha256(clock),accepted_packing_manifest_sha256=file_sha256(manifest),
        packing_fingerprint=json.loads(manifest.read_text())['fingerprint'],legacy_inventory_plan_sha256=file_sha256(old),
        model_inventory=bodies,complete_numerical_proofs=qualification,
        recurrent_updates=r_end,recurrent_targets=targets(r_end),ordinary_work_updates=o_end,ordinary_work_targets=targets(o_end),
        equal_data_updates=r_end,stable_trunk_update=r_stable,ordinary_work_stable_update=o_stable,work_match=match,
        main_required_seconds=24*3600,measured_warm_forecast_seconds=warm,
        phase_limits=dict(recurrent=14*3600,ordinary_trunk=8*3600,ordinary_equal=2*3600,ordinary_finish=4*3600),
        main_quality_accessed=False,main_fit_admitted=False,terminal_alpha_spent=0,
        scope_change='New prospective scientific exposure choice;clock amendment did not retroactively change an old endpoint.')
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'scope.json',value);write_state(PUBLIC/'scope.json',value)
    return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(freeze(a.root),indent=2))
