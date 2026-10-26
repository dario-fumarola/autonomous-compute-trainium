"""One bounded cross-fitted predictability probe; no language-model changes."""
import argparse
import json
import math
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np

from compute_trainium.prefix_value_probe import lag_features, crossfit_probe, group_fold
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY = 'past-surprisal-value-probe-v1'
PUBLIC = Path('results/future_span_depth_oracle_20261025/token/prefix-probe')


def read(path):
    return json.loads(Path(path).read_text())


def register(root):
    out = root/IDENTITY
    if out.exists() or PUBLIC.exists(): raise FileExistsError('Preserve registered probes')
    parent = root/'sampled-token-depth-oracle-v1'
    result = read(parent/'report.json')
    assert result['status'] == 'COMPLETE_RETROSPECTIVE_TOKEN_OUTPUT_SELECTION_BOUND'
    paths = ['scripts/probe_prefix_surprisal_value.py', 'compute_trainium/prefix_value_probe.py',
             'tests/test_prefix_value_probe.py']
    plan = dict(identity=IDENTITY, registered_unix=time.time(), source_sha256={p: file_sha256(p) for p in paths},
        parent_report_sha256=file_sha256(parent/'report.json'), input_hashes=result['input_hashes'],
        seeds=[5231, 7283], depths=[1, 2, 3], source_group_folds=5,
        fold_hash='SHA256(past-surprisal-probe-v1\\0 + raw source group),first8 bytes big endian,mod5',
        context=1024, past_loss_clip=20., ridge=1e-3,
        features=['previous depth-one loss', 'mean preceding up-to4 depth-one losses',
                  'mean preceding up-to16 depth-one losses', 'position within window /1023', 'has prior target'],
        fit='Weighted training-fold feature normalization;intercept unpenalized;fixed linear ridge predicts depth1/2 loss differences to depth3.',
        weights={'web': .6, 'math': .2, 'code': .2}, seconds_limit=180,
        only_one_feature_family_and_penalty=True, coefficient_fits=10,
        advancement_rule='A point improvement above0.01 over constant depth3 in BOTH base seeds may motivate further diagnosis after declared controls finish. No routing campaign is automatically admitted. Failure closes this cheap feature family,not all prefix selectors.',
        current_target_loss_in_features=False, other_depth_loss_in_features=False,
        group_or_domain_or_document_ids_in_features=False,
        language_model_forwards=0, language_model_updates=0, accelerator_chips=0,
        test_accessed=False, terminal_alpha_spent=False, native_routing_campaign_admitted=False,
        scope='Retrospective statistical probe on consumed DEV,selected after oracle and uniform-mixture results. Fit separate low-capacity readout regressions on other source groups and evaluate each held-out group. Only PREVIOUS observed depth-one token surprisals in the SAME1024-position window and prefix position are features;reset at document/window boundaries. Current/future target losses are labels only. Prior depth-one probabilities would require an inference head and their cost must be charged in any future method. This predicts selections among frozen uniform-depth distributions,not mixed-state routing. Cross-fitted policies are an analysis device,not one deployed/pretrained model. No external teacher,generated text,language-model updates,fresh panel,TEST,alpha,4B or new native work. No hardware efficiency or terminal claim;ordinary8 remains the strong benchmark.')
    out.mkdir(); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'registration.json', plan)
    return plan


def weighted(values, frames, weights):
    answer = {}; total = {}
    for domain in weights:
        group = [f for f in frames if f['domain'] == domain]
        total[domain] = sum(f['length'] for f in group)
        answer[domain] = math.fsum(float(values[f['offset']:f['offset']+f['length']].sum()) for f in group)/total[domain]
    return math.fsum(weights[d]*answer[d] for d in weights), answer


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args(); out = args.root/IDENTITY
    plan = read(out/'plan.json') if (out/'plan.json').exists() else register(args.root)
    if args.prepare_only: return
    if (out/'started.json').exists(): raise FileExistsError('Preserve every stopped/completed probe')
    started = time.time(); cpu = time.process_time(); plan_sha = file_sha256(out/'plan.json')
    write_state(out/'started.json', dict(unix=started, plan_sha256=plan_sha))
    parent = args.root/'sampled-token-depth-oracle-v1'; result = dict(status='STOPPED_PAST_SURPRISAL_PROBE')
    def guard():
        if time.time()-started > plan['seconds_limit']: raise TimeoutError('Bounded statistical probe')
        if file_sha256(out/'plan.json') != plan_sha or any(file_sha256(p) != h for p, h in plan['source_sha256'].items()): raise AssertionError('Frozen probe source changed')
        if file_sha256(parent/'report.json') != plan['parent_report_sha256']: raise AssertionError('Frozen parent result changed')
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available < 544*1024**3 or min(shutil.disk_usage(args.root).free, shutil.disk_usage('/').free) < 35*1024**3: raise RuntimeError('Protected probe resources')
    try:
        guard(); parent_result = read(parent/'report.json'); records = []
        for seed in plan['seeds']:
            frames = read(parent/f's{seed}-d1/token-loss-index.json')['records']; arrays = []
            for depth in plan['depths']:
                expected = next(r for r in plan['input_hashes'] if r['seed'] == seed and r['depth'] == depth)
                folder = parent/f's{seed}-d{depth}'
                if file_sha256(folder/'token-losses.npy') != expected['token_losses_sha256'] or file_sha256(folder/'token-loss-index.json') != expected['index_sha256']: raise AssertionError('Original paired loss arrays changed')
                if read(folder/'token-loss-index.json')['records'] != frames: raise AssertionError('Original target order changed')
                arrays.append(np.load(folder/'token-losses.npy', mmap_mode='r'))
            losses = np.stack(arrays); features = lag_features(losses[0], frames, plan['context'], plan['past_loss_clip'])
            guard()
            fit = crossfit_probe(features, losses, frames, folds=plan['source_group_folds'], ridge=plan['ridge'], weights=plan['weights'])
            guard(); columns = np.arange(losses.shape[1]); chosen = losses[fit['choices'], columns]
            oracle = losses.min(axis=0)
            assert np.all(chosen >= oracle-1e-12)
            nll, domains = weighted(chosen, frames, plan['weights'])
            constant_nll, _ = weighted(losses[fit['constant_choices'], columns], frames, plan['weights'])
            best, _ = weighted(losses[2], frames, plan['weights']); minimum, _ = weighted(oracle, frames, plan['weights'])
            old = next(r for r in parent_result['records'] if r['seed'] == seed)
            assert abs(minimum-old['token_oracle_nll']) < 1e-12 and abs(best-old['constant_depth_nll']['3']) < 1e-12
            records.append(dict(seed=seed, cross_fitted_selection_nll=nll, domain_nll=domains,
                constant_depth3_nll=best, gain_over_constant_depth3=best-nll,
                training_fold_chosen_constant_nll=constant_nll,
                gain_over_training_only_constant=constant_nll-nll, token_oracle_nll=minimum,
                remaining_oracle_gap=nll-minimum,
                chosen_target_counts={str(d): int((fit['choices'] == i).sum()) for i, d in enumerate(plan['depths'])},
                source_groups_by_fold={str(i): len({f['group'] for f in frames if group_fold(f['group']) == i}) for i in range(plan['source_group_folds'])},
                fitted_regression_coefficients=sum(r['fitted_coefficients'] for r in fit['fit_records']), fit_records=fit['fit_records']))
        result = dict(status='COMPLETE_CROSS_FITTED_PAST_SURPRISAL_PROBE', records=records,
            mean_selection_nll=float(np.mean([r['cross_fitted_selection_nll'] for r in records])),
            mean_gain_over_constant_depth3=float(np.mean([r['gain_over_constant_depth3'] for r in records])),
            both_seed_point_gains_exceed_001=all(r['gain_over_constant_depth3'] > .01 for r in records),
            native_routing_campaign_admitted=False, deployable_inference_cost_established=False,
            terminal_claim=False, scope=plan['scope'])
    except BaseException as error:
        result.update(error=str(error), error_kind=type(error).__name__, traceback=traceback.format_exc())
        write_state(out/'initiating-failure.json', result); raise
    finally:
        result.update(plan_sha256=plan_sha, wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            language_model_forwards=0, language_model_updates=0, assigned_chips=0,
            test_accessed=False, terminal_alpha_spent=False)
        write_state(out/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
