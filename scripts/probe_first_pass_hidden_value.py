"""Bounded TRAIN-to-consumed-DEV frozen hidden-state information diagnostic."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('USE_TORCH_XLA', '0')
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from compute_trainium.data import document_windows
from compute_trainium.future_span_evaluation import prediction_state, window_state
from compute_trainium.hidden_value_probe import (prefix_features, fit_readouts,
    predict_readouts, allocation_curve, domain_weights, matched_curve_nll)
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN

IDENTITY = 'first-pass-hidden-value-probe-v1'
PUBLIC = Path('results/first_pass_hidden_value_20261026')
DOMAINS = {'web': 1, 'math': 2, 'code': 3}


def read(path):
    return json.loads(Path(path).read_text())


def guard(root, started, limit):
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available < 544*1024**3 or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3:
        raise RuntimeError('Protected CPU information-diagnostic resources')
    if time.time()-started > limit:
        raise TimeoutError('Bounded CPU information diagnostic')


def register(root):
    out = root/IDENTITY
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every probe attempt')
    parent = root/'sampled-token-depth-oracle-v1'
    old = read(parent/'plan.json'); result = read(parent/'report.json')
    main = root/'future-span-main-v2/plan.json'; science = read(main)
    audit = root/'natural-append-6500m-v2/independent-audit.json'
    checked = read(audit)
    assert checked['status'] == 'PASS_INDEPENDENT_SECOND_EXTENSION_AUDIT'
    assert checked['protected_group_overlap'] == checked['exact_id_overlap'] == 0
    assert result['all_original_document_depth_scores_reproduced']
    files = ['scripts/probe_first_pass_hidden_value.py', 'compute_trainium/hidden_value_probe.py',
        'tests/test_hidden_value_probe.py', 'scripts/audit_parcae_precise_training_point.py',
        'scripts/qualify_parcae_candidate_cpu.py', 'compute_trainium/future_span_evaluation.py',
        'compute_trainium/training.py', 'compute_trainium/data.py']
    jobs = [j for j in old['jobs'] if j['depth'] == 1]
    start = 1920*512
    plan = dict(identity=IDENTITY, registered_unix=time.time(), seeds=[5231, 7283],
        source_sha256={p: file_sha256(p) for p in files}, jobs=jobs,
        parent_plan_sha256=file_sha256(parent/'plan.json'), parent_report_sha256=file_sha256(parent/'report.json'),
        scientific_plan_sha256=file_sha256(main), data=science['data'],
        train_source_isolation_audit_sha256=file_sha256(audit),
        train_first_window=start, train_windows=512, fitting_windows=384,
        calibration_windows=128, train_positions_follow_original_checkpoint_cursor=True,
        feature_columns=['768 first-core-pass state coordinates before projection/coda',
                         'current first-pass prediction entropy', 'top1-minus-top2 probability margin', 'window position /1023'],
        feature_storage='float16 private cache;original FP32 values checked for relative quantization error<=0.001',
        feature_width=771, quantization_relative_l2_limit=.001,
        loss_labels='FP64 loss(depth1)-loss(depth2) on genuine source next tokens;no target tokens/losses in predictors',
        fit=dict(ridge=.01, epochs=8, hidden_width=64, batch_size=2048, seed=31415),
        requested_extra_pass_fractions=[0., .1, .25, .5, .75, .9, 1.], primary_fraction=.5,
        threshold_policy='Quantile of unlabeled final128 TRAIN-window scores;apply independently per DEV position.No held-out global top-k.',
        comparison='Report realized fractions. Random comparator matches each policy expected token fraction. Entropy cost interpolation uses DEV feature frequencies only and is explicitly retrospective,not a freshly calibrated policy.',
        advancement_rule='Primary MLP at TRAIN-calibrated half allocation must save>0.01 nats against both same-fraction expected random and descriptive cost-matched entropy in BOTH base seeds to motivate actual routed-semantics preparation.No native work admitted automatically.',
        controls=['constant supported depth1', 'constant supported depth2', 'same-fraction expected random',
                  'TRAIN-thresholded current entropy', 'one fixed linear readout'],
        max_parallel_cpu_workers=2, worker_seconds_limit=2400, total_seconds_limit=3000,
        anticipated_window_forwards=2*(512*2+1050), projected_private_cache_bytes=2*(512*1024+754750)*(771*2+20),
        measured_prior_token_replay_seconds=result['wall_seconds'], measured_prior_window_forwards=6300,
        projected_trace_seconds_with_2x_allowance=result['wall_seconds']*(2*(512*2+1050))/6300*2,
        language_model_updates=0, assigned_chips=0, test_accessed=False, terminal_alpha_spent=0,
        native_routing_campaign_admitted=False,
        scope='Exploratory frozen-model information test on TRAIN-source traces and already consumed source-disjoint DEV. Same original sampled-depth checkpoints support both exits. Train windows are the next512 beyond their1.005B cursor;fit on first384,calibrate unlabeled thresholds on final128. Source isolation relies on published whole-corpus independent exact/group audit;its approximate-near-duplicate limitation persists. One fixed linear and one small nonlinear readout per seed;no DEV fitting,hyperparameter search,new LM objective or LM update. Offline selection among uniform-depth predictions is not execution with mixed KV states,no actual FLOP/latency saving or broad efficiency claim. Current ordinary8 has1.955x these checkpoints training work and a different second seed;not a matched-budget comparator. Broad NLL/0.01/strongest eligible ordinary criterion unchanged;no4B,reserve,TEST or confirmation.')
    if plan['projected_trace_seconds_with_2x_allowance']+600 > plan['total_seconds_limit']:
        raise RuntimeError('Measured trace projection plus bounded probe fitting does not fit')
    if shutil.disk_usage(root).free-plan['projected_private_cache_bytes'] < 36*1024**3:
        raise RuntimeError('Prospective full trace cache leaves insufficient reserve and margin')
    out.mkdir(); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'registration.json', plan)
    return plan


def worker(root, plan, seed):
    started = time.time(); cpu = time.process_time()
    torch.set_num_threads(12); torch.set_num_interop_threads(1)
    out = root/IDENTITY/f's{seed}'; out.mkdir(exist_ok=False)
    job = next(j for j in plan['jobs'] if j['seed'] == seed)
    checkpoint = root/job['checkpoint']
    if file_sha256(checkpoint) != job['checkpoint_sha256']:
        raise AssertionError('Frozen checkpoint changed')
    upstream = root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PIN
            or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Clean pinned independent model reference required')
    sys.path.insert(0, str(upstream))
    science = read(root/'future-span-main-v2/plan.json'); case = science['cases'][job['case']]
    if any(file_sha256(p) != h for p, h in science['source_sha256'].items()):
        raise AssertionError('Original scientific implementation changed')
    config = ParcaeCandidateConfig(**science['model'])
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['update'] != 1920 or saved['cursor'] != plan['train_first_window'] or saved['case_id'] != job['case']:
        raise AssertionError('Checkpoint identity/source cursor differs')
    if digest({'model': dict(sorted(saved['model'].items())), 'optimizer': saved['optimizer']}) != saved['model_optimizer_digest']:
        raise AssertionError('Complete original checkpoint readback differs')
    model = make_reference(config, True).eval()
    model.load_state_dict(prediction_state(saved['model']), strict=True); del saved
    before = digest(dict(sorted(model.state_dict().items())))
    current = None; captured = []
    model.initialize_state = lambda _: current
    hook = model.transformer.core_block[-1].register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    if file_sha256(root/plan['data']['packed']/'manifest.json') != plan['data']['manifest_sha256']:
        raise AssertionError('Original packed TRAIN manifest changed')
    packed = PackedCorpus.from_mapped(root/plan['data']['packed'], 1024, 23, plan['data']['corpus_sha256'])
    train_count = int(sum(packed.counts(plan['train_first_window'], plan['train_windows'])))
    fit_count = int(sum(packed.counts(plan['train_first_window'], plan['fitting_windows'])))
    # Header/cache allocation is included in the prospective disk budget.
    train_x = np.lib.format.open_memmap(out/'train-features.npy', mode='w+', dtype=np.float16, shape=(train_count, 771))
    train_l = np.lib.format.open_memmap(out/'train-losses.npy', mode='w+', dtype=np.float64, shape=(2, train_count))
    train_d = np.lib.format.open_memmap(out/'train-domains.npy', mode='w+', dtype=np.uint8, shape=(train_count,))
    dev_x = np.lib.format.open_memmap(out/'dev-features.npy', mode='w+', dtype=np.float16, shape=(754750, 771))
    quant_error = quant_norm = 0.; forwards = 0; original_loss_max_error = 0.

    def evaluate(tokens, initial, depth, want_features):
        nonlocal current, forwards, quant_error, quant_norm
        guard(root, started, plan['worker_seconds_limit'])
        current = initial; captured.clear()
        logits = model(tokens, return_logits=True, num_steps_pair=torch.tensor([depth-1, 1]))['logits'][0]
        forwards += 1
        if len(captured) != depth or not torch.isfinite(logits).all():
            raise AssertionError('Reference state capture/finite logits differ')
        features = None
        if want_features:
            raw = prefix_features(captured[0][0], logits)
            quantized = raw.half()
            if not torch.isfinite(quantized).all():
                raise AssertionError('Nonfinite private feature quantization')
            quant_error += float((raw.double()-quantized.double()).square().sum())
            quant_norm += float(raw.double().square().sum())
            features = quantized.numpy().copy()
        terms = []
        for left in range(0, len(logits), 128):
            values = logits[left:left+128].double()
            terms.append(values.logsumexp(-1).numpy())
        # Target-specific gather is deliberately outside prefix_features.
        return logits, np.concatenate(terms), features

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        cursor = 0
        for ordinal in range(plan['train_windows']):
            absolute = plan['train_first_window']+ordinal
            batch, domains = packed.batch([absolute], 0)
            mask = batch.target_mask[0].bool().numpy(); count = int(mask.sum())
            initial = window_state(config, absolute, case['state_seed'], 1024)
            for depth in (1, 2):
                logits, normalizer, features = evaluate(batch.inputs, initial, depth, depth == 1)
                target = batch.targets[0]
                losses = normalizer-logits.double().gather(-1, target[:, None]).squeeze(-1).numpy()
                train_l[depth-1, cursor:cursor+count] = losses[mask]
                if depth == 1:
                    train_x[cursor:cursor+count] = features[mask]
                    train_d[cursor:cursor+count] = domains[0].numpy()[mask]
            cursor += count
            if (ordinal+1) % 32 == 0:
                write_state(out/'progress.json', dict(phase='TRAIN traces', windows=ordinal+1, forwards=forwards, wall_seconds=time.time()-started))
        if cursor != train_count:
            raise AssertionError('Original TRAIN target mask coverage differs')
        source = root/science['evaluation']['corpus']
        if file_sha256(source) != science['evaluation']['corpus_sha256']:
            raise AssertionError('Consumed DEV source changed')
        docs = sorted([d for d in read(source) if d['split'] == 'dev' and d['cohort'] == 'curve'], key=lambda d: d['id'])
        parent = root/'sampled-token-depth-oracle-v1'/f's{seed}-d1'
        frames = read(parent/'token-loss-index.json')['records']
        expected = np.load(parent/'token-losses.npy', mmap_mode='r')
        dev_domains = np.empty(len(expected), dtype=np.uint8)
        cursor = 0; dev_windows = 0
        for doc, frame in zip(docs, frames):
            if (doc['id'], doc['group'], doc['domain'], len(doc['tokens']), cursor) != (frame['id'], frame['group'], frame['domain'], frame['length'], frame['offset']):
                raise AssertionError('Original DEV source/index order differs')
            for window, (tokens, mask) in enumerate(document_windows(doc['tokens'], 1024)):
                count = int(mask.sum())
                initial = window_state(config, window, case['state_seed'], count)
                logits, normalizer, features = evaluate(tokens[None, :count], initial, 1, True)
                losses = normalizer-logits.double().gather(-1, tokens[1:count+1, None]).squeeze(-1).numpy()
                original_loss_max_error = max(original_loss_max_error, float(np.abs(losses-expected[cursor:cursor+count]).max()))
                dev_x[cursor:cursor+count] = features
                dev_domains[cursor:cursor+count] = DOMAINS[doc['domain']]
                cursor += count; dev_windows += 1
            if dev_windows % 50 < 3:
                write_state(out/'progress.json', dict(phase='DEV first-pass features', windows=dev_windows, forwards=forwards, wall_seconds=time.time()-started))
        if cursor != 754750 or len(docs) != len(frames) or dev_windows != 1050 or original_loss_max_error > 1e-9:
            raise AssertionError('Original held-out first-pass scores/coverage not reproduced')
    hook.remove()
    if digest(dict(sorted(model.state_dict().items()))) != before:
        raise AssertionError('Frozen feature readout changed model weights')
    del model, packed
    for array in (train_x, train_l, train_d, dev_x): array.flush()
    relative = (quant_error/max(quant_norm, 1e-30))**.5
    if relative > plan['quantization_relative_l2_limit']:
        raise AssertionError('Declared feature quantization bound failed')
    guard(root, started, plan['worker_seconds_limit'])
    trace_seconds = time.time()-started
    write_state(out/'progress.json', dict(phase='TRAIN statistical fits', forwards=forwards, wall_seconds=trace_seconds))
    fitted = fit_readouts(train_x[:fit_count], train_l[0, :fit_count]-train_l[1, :fit_count],
        train_d[:fit_count], **plan['fit'])
    calibration = predict_readouts(fitted, train_x[fit_count:])
    predictions = predict_readouts(fitted, dev_x)
    # Confidence baselines use the same private float16 feature cache.
    calibration['entropy'] = np.array(train_x[fit_count:, -3], dtype=np.float64)
    predictions['entropy'] = np.array(dev_x[:, -3], dtype=np.float64)
    losses = []
    for depth in (1, 2):
        folder = root/'sampled-token-depth-oracle-v1'/f's{seed}-d{depth}'
        expected_hash = next(h for h in read(root/'sampled-token-depth-oracle-v1/report.json')['input_hashes'] if h['seed'] == seed and h['depth'] == depth)
        if file_sha256(folder/'token-losses.npy') != expected_hash['token_losses_sha256']:
            raise AssertionError('Original loss-label arrays changed')
        losses.append(np.load(folder/'token-losses.npy', mmap_mode='r'))
    losses = np.stack(losses)
    curves = {key: allocation_curve(calibration[key], values, losses, dev_domains,
        plan['requested_extra_pass_fractions']) for key, values in predictions.items()}
    primary = next(row.copy() for row in curves['mlp'] if row['requested_train_fraction'] == plan['primary_fraction'])
    entropy_reference = matched_curve_nll(curves['entropy'], primary['realized_token_fraction'])
    primary['descriptive_cost_matched_entropy_nll'] = entropy_reference
    primary['gain_over_descriptive_cost_matched_entropy'] = entropy_reference-primary['nll']
    w = domain_weights(dev_domains)
    normalized_oracle = float(w @ losses.min(0))
    private_hashes = {p.name: file_sha256(p) for p in out.glob('*.npy')}
    report = dict(status='COMPLETE_TRAIN_SOURCE_TO_CONSUMED_DEV_HIDDEN_VALUE_PROBE', seed=seed,
        checkpoint_sha256=job['checkpoint_sha256'], source_isolation='Original independent whole-corpus TRAIN/held-out exact/group audit;no fresh reserve read',
        train_targets=train_count, fitting_targets=fit_count, calibration_targets=train_count-fit_count,
        dev_targets=cursor, dev_source_groups=len({f['group'] for f in frames}),
        original_depth_one_max_position_loss_error=original_loss_max_error,
        feature_quantization_relative_l2=relative,
        constant_depth_nll={'1': float(w @ losses[0]), '2': float(w @ losses[1])},
        two_depth_target_informed_oracle_nll=normalized_oracle,
        primary=primary, curves=curves,
        passes_information_screen=primary['gain_over_same_fraction_random'] > .01 and primary['gain_over_descriptive_cost_matched_entropy'] > .01,
        statistical_optimizer_updates=fitted['statistical_optimizer_updates'], fitted_readout_parameters=fitted['fitted_readout_parameters'],
        language_model_window_forwards=forwards, language_model_updates=0, assigned_chips=0,
        trace_wall_seconds=trace_seconds, wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        private_trace_hashes=private_hashes,
        deployable_routing_or_cost_established=False, test_accessed=False, terminal_alpha_spent=0,
        scope=plan['scope'])
    guard(root, started, plan['worker_seconds_limit'])
    write_state(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--worker-seed', type=int)
    args = parser.parse_args(); out = args.root/IDENTITY
    plan = read(out/'plan.json') if (out/'plan.json').exists() else register(args.root)
    if args.prepare_only: return
    for path, expected in plan['source_sha256'].items():
        if file_sha256(path) != expected: raise AssertionError('Registered probe implementation changed')
    if file_sha256(args.root/'future-span-main-v2/plan.json') != plan['scientific_plan_sha256']:
        raise AssertionError('Original parent scientific plan changed')
    if file_sha256(args.root/'natural-append-6500m-v2/independent-audit.json') != plan['train_source_isolation_audit_sha256']:
        raise AssertionError('Original whole-corpus source-isolation audit changed')
    for path, expected in [('sampled-token-depth-oracle-v1/plan.json', plan['parent_plan_sha256']),
                           ('sampled-token-depth-oracle-v1/report.json', plan['parent_report_sha256'])]:
        if file_sha256(args.root/path) != expected:
            raise AssertionError('Frozen original diagnostic input changed')
    if args.worker_seed is not None:
        if args.worker_seed not in plan['seeds']: raise ValueError('Unregistered seed')
        worker(args.root, plan, args.worker_seed); return
    if (out/'started.json').exists(): raise FileExistsError('Preserve stopped/completed attempt')
    started = time.time(); plan_sha = file_sha256(out/'plan.json')
    write_state(out/'started.json', dict(unix=started, plan_sha256=plan_sha))
    processes = []; result = dict(status='STOPPED_HIDDEN_VALUE_PROBE')
    def interrupted(number, frame):
        raise KeyboardInterrupt('Bounded CPU probe interrupted: '+str(number))
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        guard(args.root, started, plan['total_seconds_limit'])
        for seed in plan['seeds']:
            log = (out/f's{seed}.log').open('x')
            p = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'scripts.probe_first_pass_hidden_value',
                '--root', str(args.root), '--worker-seed', str(seed)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close(); processes.append((seed, p))
        while any(p.poll() is None for _, p in processes):
            guard(args.root, started, plan['total_seconds_limit'])
            if any(p.poll() not in (None, 0) for _, p in processes):
                raise RuntimeError('Registered CPU hidden-value worker failed;see preserved log')
            time.sleep(1)
        if any(p.returncode for _, p in processes): raise RuntimeError('CPU probe worker failed')
        records = [read(out/f's{s}/report.json') for s in plan['seeds']]
        result = dict(status='COMPLETE_TRAIN_SOURCE_HIDDEN_VALUE_INFORMATION_DIAGNOSTIC', records=records,
            both_seed_information_gates_pass=all(r['passes_information_screen'] for r in records),
            language_model_window_forwards=sum(r['language_model_window_forwards'] for r in records),
            statistical_optimizer_updates=sum(r['statistical_optimizer_updates'] for r in records),
            native_routing_campaign_admitted=False, scope=plan['scope'])
    except BaseException as error:
        result.update(error=str(error), error_kind=type(error).__name__, traceback=traceback.format_exc())
        write_state(out/'initiating-failure.json', result)
        raise
    finally:
        for _, p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
        for number, handler in handlers.items(): signal.signal(number, handler)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(plan_sha256=plan_sha, wall_seconds=time.time()-started,
            child_cpu_seconds=usage.ru_utime+usage.ru_stime, language_model_updates=0,
            assigned_chips=0, test_accessed=False, terminal_alpha_spent=0, terminal_success=False,
            cost_scope='CPU-only statistical experiment;CPU model forwards and probe fits counted separately.No accelerator jobs;reserved-instance opportunity is not productive chip computation.')
        write_state(out/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'wall_seconds', 'assigned_chips')}))


if __name__ == '__main__': main()
