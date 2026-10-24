"""Declared descriptive contrasts after all common-FP32 independent audits."""
import argparse
import json
import math
import time
from pathlib import Path

from compute_trainium.model import ModelConfig
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


WEIGHTS = {'web': .6, 'math': .2, 'code': .2}


def checked_metric(record):
    if (not record['copy_checksum_and_full_state_readback']
            or not math.isfinite(record['cpu_minus_native_nll'])
            or abs(record['cpu_minus_native_nll']) > .0002):
        raise ValueError('Independent scoring/full-state-copy gate failed')
    metrics = record['metrics']
    if metrics['weights'] != WEIGHTS or set(metrics['domains']) != set(WEIGHTS):
        raise ValueError('Incorrect domain weights/coverage')
    value = 0.
    for domain, weight in WEIGHTS.items():
        row = metrics['domains'][domain]
        if row['real_targets'] <= 0:
            raise ValueError('Empty domain')
        nll = row['nll_sum']/row['real_targets']
        if not math.isfinite(nll) or not math.isfinite(row['nll']) or abs(nll-row['nll']) > 1e-10:
            raise ValueError('Inconsistent domain loss')
        value += weight*nll
    if not math.isfinite(metrics['weighted_nll']) or abs(value-metrics['weighted_nll']) > 1e-10:
        raise ValueError('Inconsistent weighted NLL')
    return value


def analyze(root, public):
    plan = json.loads((public/'plan.json').read_text())
    private = root/plan['identity']
    terminal = private/'report.json'
    if not terminal.exists():
        return False
    terminal = json.loads(terminal.read_text())
    if terminal['status'] != 'COMPLETE_COMMON_FP32_DEPTH_CONNECTION_COMPARISON':
        write_state(public/'analysis-status.json', {'status': 'UNAVAILABLE_STOPPED_CAMPAIGN',
            'campaign_status': terminal['status'], 'scope': 'No completed comparison is imputed.'})
        return True
    registration = json.loads((public/'analysis-registration.json').read_text())
    for path, sha in registration['source_sha256'].items():
        if file_sha256(path) != sha:
            raise AssertionError('Readout changed after registration')
    for path, sha in plan['source_sha256'].items():
        if file_sha256(path) != sha:
            raise AssertionError('Scientific source changed')
    prior = Path('results/reinjection_pilot_20261024')
    old = json.loads((prior/'independent-cpu-verification.json').read_text())['records']
    new = json.loads((public/'independent-cpu-verification.json').read_text())['records']
    inventory = [('ordinary1B', 'reinjection-pilot-v1-ordinary-s419-u3840', old),
                 ('connection1B', 'reinjection-pilot-v1-reinjection-s419-u3840', old),
                 ('fixed1B', 'depth-connection-v1-fixed-s419-u3840', new),
                 ('ordinaryMatched', 'depth-connection-v1-ordinary-s419-u6496', new),
                 ('connectionMatched', 'depth-connection-v1-reinjection-s419-u6496', new)]
    rows = {}
    panel = None
    for label, identity, records in inventory:
        matches = [r for r in records if r['identity'] == identity]
        if len(matches) != 1:
            raise AssertionError('Missing or duplicate independent endpoint')
        r = matches[0]
        nll = checked_metric(r)
        counts = {d: r['metrics']['domains'][d]['real_targets'] for d in WEIGHTS}
        if panel is None: panel = counts
        if counts != panel or r['seed'] != 419:
            raise AssertionError('Unpaired source targets/seed')
        report = json.loads((root/'runs'/identity/'report.json').read_text())
        sig = report['signature']
        expected_targets = plan['targets']['6496' if label.endswith('Matched') else '3840']
        if (r['training_real_targets'] != expected_targets or report['total_real_targets'] != expected_targets
                or report['initial_model_digest'] != plan['initial_main_digest']
                or report['precision'] != 'fp32' or not report['all_replica_model_optimizer_states_equal']
                or sig['dev_corpus_sha256'] != plan['dev_sha256']):
            raise AssertionError('Initialization, precision, exposure, panel or replicas differ')
        model = ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
        if model.parameter_count() != 150436608 or report['parameters'] != 150436608:
            raise AssertionError('Parameter mismatch')
        if report['schedule']['decay_end'] != expected_targets:
            raise AssertionError('Endpoint did not complete its own decay')
        config = PreNormRecurrenceConfig(**{**{k:getattr(model,k) for k in ModelConfig.__dataclass_fields__},
                                             'repetitions': 2 if label=='fixed1B' else 1})
        rows[label] = {'identity': identity, 'nll': nll,
            'domain_nll': {d:r['metrics']['domains'][d]['nll'] for d in WEIGHTS},
            'training_real_targets': expected_targets, 'parameters': report['parameters'],
            'cumulative_estimated_training_matmuls': report['total_training_matmul_flops'],
            'inference_full1024_window_matmuls': config.flops(1024, 1024)['forward_matmul_flops'],
            'checkpoint_sha256': r['checkpoint_sha256'],
            'independent_native_cpu_nll_difference': r['cpu_minus_native_nll']}
    fixed = rows['fixed1B']
    contrasts = {}
    for label, baseline in rows.items():
        if label == 'fixed1B': continue
        ratio = fixed['cumulative_estimated_training_matmuls']/baseline['cumulative_estimated_training_matmuls']
        if label.endswith('Matched') and abs(ratio-1) > .0002:
            raise AssertionError('Registered estimated training-work match differs')
        contrasts[label] = {'baseline_minus_fixed_nll': baseline['nll']-fixed['nll'],
            'domain_gains': {d:baseline['domain_nll'][d]-fixed['domain_nll'][d] for d in WEIGHTS},
            'fixed_over_baseline_estimated_training_matmuls': ratio,
            'interpretation': 'Positive favors fixed; one development pair, no seed interval or significance.'}
    summary = {'status': 'COMPLETE_AUDITED_COMMON_FP32_DEPTH_CONNECTION_READOUT',
        'seed': 419, 'models': rows, 'contrasts': contrasts, 'dev_domain_targets': panel,
        'test_accessed': False, 'new_terminal_attempt': False,
        'scope': 'One reused consumed-DEV pair. Same FP32-from-initialization recipe and original weights/source order. Estimated matmuls include head/backward; hardware time, optimizer work and parent campaign costs remain separately reported. No significance, mechanism universality or goal completion.'}
    write_state(public/'summary.json', summary)
    lines = ['# Common-FP32 depth/connection result', '',
        'One descriptive seed419 pair on consumed DEV; independent full-panel',
        'FP32 scores and full-state-copy audits pass. No TEST or terminal claim.', '',
        '| Model | Real targets | DEV NLL | Estimated training matmuls |',
        '| --- | ---: | ---: | ---: |']
    for label,r in rows.items():
        lines.append(f"| {label} | {r['training_real_targets']:,} | {r['nll']:.9f} | {r['cumulative_estimated_training_matmuls']:.6e} |")
    lines += ['', 'Positive differences below favor fixed recurrence:', '']
    for label,r in contrasts.items():
        lines.append(f"- Versus {label}: {r['baseline_minus_fixed_nll']:+.9f} nats/token.")
    lines += ['', 'Each endpoint completes its own decay. Stronger connection/ordinary',
        'controls spend matching estimated matmuls on more natural source text.',
        'This is a development comparison, not evidence across training seeds.',
        'The inexpensive connection is retained; no gain is attributed to another',
        'core execution by subtracting models with different precision histories.', '',
        '[Exact metrics](summary.json) · [Current execution costs](execution-costs.json) ·',
        '[Reused parent campaign costs](../reinjection_pilot_20261024/execution-costs.json).', '',
        'Current receipts include discarded LR trials and audits; reused parent work',
        'must not be forgotten. Overlapping intervals are not sequential node time.',
        'Nominal matmul matching is not matched elapsed time or inference latency.']
    (public/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_state(public/'analysis-status.json', {'status': summary['status'], 'updated_unix':time.time()})
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--follow', action='store_true')
    a = p.parse_args()
    while not analyze(a.root, a.output):
        plan = json.loads((a.output/'plan.json').read_text())
        if time.time() > plan['hard_deadline_unix']+300:
            raise TimeoutError('Scientific campaign has no terminal record after its deadline')
        if not a.follow: return
        time.sleep(30)


if __name__ == '__main__':
    main()
