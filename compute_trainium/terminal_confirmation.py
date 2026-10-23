"""Future fixed-sample confirmation, separate from the frozen depth study.

These functions do not allocate an attempt, consume a panel or inspect a model.
A preregistered experiment and an independently verified data gate are still
required. Seed intervals are conditional on the fixed evaluation panel.
"""
import math
import statistics


def allocated_alpha(attempt):
    """Two-sided family allocation; sum over all positive attempts is 0.05."""
    if type(attempt) is not int or attempt < 1 or attempt > 1000:
        raise ValueError('Attempt must be a positive integer no greater than 1000')
    return math.ldexp(.05, -attempt)


def validate_registration(spec, history, *, used_seeds, used_panel_shas):
    """Reject missing freezes, reused identities and reset alpha accounting.

    History includes every registered attempt, successful or otherwise. Hashes
    bind the declarations; they cannot prove source-group isolation or that a
    panel was not inspected. The corpus audit and release gate must do that.
    """
    attempts = [row['attempt'] for row in history]
    if attempts != list(range(1, len(history) + 1)):
        raise ValueError('History must retain every attempt in order')
    for row in [*history, spec]:
        if row['alpha'] != allocated_alpha(row['attempt']):
            raise ValueError('Alpha must follow the ongoing allocation')
    if spec['attempt'] != len(history) + 1:
        raise ValueError('Attempt numbering cannot restart or skip')
    if spec['primary_comparisons'] != 1:
        raise ValueError('This helper supports exactly one frozen primary comparison')
    for key in ['candidate_sha256', 'control_sha256', 'protocol_sha256',
                'analysis_sha256', 'panel_sha256']:
        value = spec[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid {key}')
    seeds = spec['paired_seeds']
    if len(seeds) < 5 or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('At least five distinct paired seeds are required')
    consumed_seeds = set(used_seeds).union(*(row['paired_seeds'] for row in history))
    consumed_panels = set(used_panel_shas) | {row['panel_sha256'] for row in history}
    if consumed_seeds.intersection(seeds) or spec['panel_sha256'] in consumed_panels:
        raise ValueError('Confirmation needs fresh seeds and a fresh source panel')
    if not math.isfinite(spec['minimum_gain_nats']) or spec['minimum_gain_nats'] < .01:
        raise ValueError('The useful-effect threshold cannot be weakened')
    return True


def fixed_sample_result(spec, gains):
    """Paired Student-t interval at the registered alpha, without interim looks.

    A gain is control NLL minus candidate NLL on the same source targets. This
    uses the usual independent, approximately normal paired-seed assumption;
    it is not a source-group confidence interval or an anytime-valid test.
    """
    from scipy.stats import t

    seeds = spec['paired_seeds']
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or set(gains) != set(seeds):
        raise ValueError('Every preregistered paired seed must be present exactly once')
    values = [gains[seed] for seed in seeds]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite or failed seeds cannot be omitted')
    alpha = allocated_alpha(spec['attempt'])
    if spec['alpha'] != alpha or spec['primary_comparisons'] != 1:
        raise ValueError('The registered primary allocation changed')
    threshold = spec['minimum_gain_nats']
    if not math.isfinite(threshold) or threshold < .01:
        raise ValueError('Invalid meaningful-effect threshold')
    mean = statistics.mean(values)
    half = float(t.isf(alpha / 2, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    lo, hi = mean - half, mean + half
    return {'attempt': spec['attempt'], 'alpha': alpha, 'paired_seed_gains_nats': gains,
            'mean_gain_nats': mean, 'interval': [lo, hi], 'minimum_gain_nats': threshold,
            'decision': ('USEFUL_GAIN_SUPPORTED' if lo > threshold else
                         'USEFUL_GAIN_RULED_OUT' if hi < threshold else 'INCONCLUSIVE'),
            'scope': 'Fixed-sample paired-seed interval conditional on this source panel; '
                     'independent source-group uncertainty and resource controls remain required.'}
