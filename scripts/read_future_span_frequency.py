"""Exploratory loss-channel diagnostic;no model forwards or scientific updates."""
import argparse
import hashlib
import json
import signal
import shutil
import time
from pathlib import Path

import numpy as np

from compute_trainium.future_span_frequency import count_source_spans, probabilities, constant_presence_loss
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, file_sha256

IDENTITY = 'future-span-frequency-diagnostic-v1'
PUBLIC = Path('results/future_span_frequency_20261025')
START = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Frequency diagnostic signal ' + str(signum))
    for signum in [signal.SIGINT, signal.SIGTERM]: signal.signal(signum, interrupted)
    main_root = args.root/'future-span-main-v2'
    main_file = main_root/'plan.json'
    parent = json.loads(main_file.read_text())
    out = args.root/IDENTITY
    plan_file = out/'plan.json'
    if not plan_file.exists():
        if (main_root/'report.json').exists():
            raise ValueError('Register before complete MAIN;this remains exploratory')
        sources = dict(parent['source_sha256'])
        for f in ['compute_trainium/future_span_frequency.py', 'scripts/read_future_span_frequency.py',
                  'tests/test_future_span_frequency.py']:
            sources[f] = file_sha256(f)
        execution = json.loads((main_root/'execution/execution-start.json').read_text())
        plan = dict(identity=IDENTITY, registered_unix=time.time(), source_sha256=sources,
            main_plan_sha256=file_sha256(main_file), fit_start=0, fit_stop=16384,
            probe_start=1920*512, probe_stop=1921*512, pseudocount=.5,
            count_seconds_limit=300, wait_deadline_unix=execution['deadline_unix'],
            future_config=parent['future_config'],
            registration_disclosure='Registered after six mature gradient reports and all first-seed MAIN losses. The observed future-head means fall from about6.70 to3.56. The frequency reference was not computed or selected from alternative sample sizes,priors or targets. Exploratory explanation only;no primary/admission change.',
            scope='Fit one constant Bernoulli frequency per token on the first16384 original TRAIN windows,with fixed Jeffreys smoothing. Score the already registered next512-window TRAIN gradient-probe targets,disjoint from that prefix. Reuse all eight original independent model objective audits only after complete MAIN. Sampled training-depth forward losses,not fixed-depth DEV predictions. No teacher,extra model training/forwards,held-out selection or success claim.')
        out.mkdir();PUBLIC.mkdir(parents=True)
        write_state(plan_file, plan);write_state(PUBLIC/'plan.json', plan)
    else:
        plan = json.loads(plan_file.read_text())
    if args.prepare_only:
        return
    if (out/'report.json').exists():
        raise FileExistsError('Preserve completed/stopped diagnostic')
    registered_sha = file_sha256(plan_file)
    def frozen():
        if (file_sha256(plan_file) != registered_sha or file_sha256(main_file) != plan['main_plan_sha256'] or
                any(file_sha256(f) != sha for f, sha in plan['source_sha256'].items())):
            raise AssertionError('Frozen source/parent changed')
    frozen()
    started = time.time()
    def guard():
        available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))
        if (available < 544*1024**3 or time.time()-started > plan['count_seconds_limit']
                or min(shutil.disk_usage(out).free, shutil.disk_usage('/').free) < 35*1024**3):
            raise RuntimeError('Frequency-count reserve or fixed CPU cap')
    guard()
    data = parent['data']
    packed = PackedCorpus.from_mapped(args.root/data['packed'], 1024, 23, data['corpus_sha256'])
    if packed.fingerprint != data['packing_fingerprint']:
        raise AssertionError('Original immutable source stream changed')
    span = plan['future_config']
    def count(first, stop):
        return count_source_spans(packed.tokens.numpy(), packed.domains.numpy(),
            start=first, stop=stop, context=span['context'], horizon=span['horizon'], stride=span['stride'],
            vocab_size=parent['model']['vocab_size'], guard=guard)
    if plan['fit_stop'] > plan['probe_start']:
        raise AssertionError('The frequency fit must not see probe targets')
    fit, size = count(plan['fit_start'], plan['fit_stop'])
    target, total = count(plan['probe_start'], plan['probe_stop'])
    prior = probabilities(fit, size)
    uniform = np.full(len(fit), (span['horizon']-1)/len(fit))
    baseline = dict(fitted_frequency_mean=constant_presence_loss(target, total, prior, span['horizon']),
        initial_constant_prior_mean=constant_presence_loss(target, total, uniform, span['horizon']),
        fit_windows=plan['fit_stop']-plan['fit_start'], fit_real_targets=sum(packed.counts(0, plan['fit_stop'])),
        fit_valid_anchors=size, probe_valid_anchors=total,
        probe_real_targets=sum(packed.counts(plan['probe_start'], 512)),
        fit_counts_sha256=hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest(),
        probe_counts_sha256=hashlib.sha256(target.astype('<i8').tobytes()).hexdigest(),
        wall_seconds=time.time()-started, assigned_chips=0, model_forwards=0, optimizer_updates=0)
    guard();np.savez(out/'private-counts.npz', fit=fit, probe=target)
    write_state(out/'frequency-reference.json', baseline)
    write_state(PUBLIC/'frequency-reference.json', baseline)
    del packed
    while not (main_root/'report.json').exists():
        frozen()
        if time.time() > plan['wait_deadline_unix']:
            raise TimeoutError('Original MAIN window expired')
        write_state(PUBLIC/'status.json', {'status':'WAITING_FOR_COMPLETE_UNCHANGED_MAIN', 'assigned_chips':0})
        time.sleep(10)
    frozen()
    finished = json.loads((main_root/'report.json').read_text())
    if (finished['status'] != 'COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT'
            or finished['plan_sha256'] != plan['main_plan_sha256']):
        raise ValueError('Full unchanged MAIN required;do not select partial successful models')
    rows, inputs = [], []
    for key, case in parent['cases'].items():
        file = main_root/'gradients'/key/'report.json'
        value = json.loads(file.read_text())
        score = next(x for x in finished['scores'] if x['case_id'] == key and x['readout'] == 'cooled1b')
        if (value['status'] != 'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                or value['checkpoint_sha256'] != score['checkpoint_sha256']
                or value['training_update'] != 1920 or value['windows'] != 512
                or value['valid_anchors'] != total or value['real_targets'] != baseline['probe_real_targets']):
            raise AssertionError('Every exact same-source mature objective audit required')
        rows.append(dict(case_id=key, cell=case['cell'], seed=case['seed'],
            next_train_batch_ntp_nll=value['cpu_ntp_nll'], future_mean=value['cpu_future_mean'],
            future_gain_over_constant_frequency=baseline['fitted_frequency_mean']-value['cpu_future_mean']))
        inputs.append(dict(case_id=key, audit_sha256=file_sha256(file), checkpoint_sha256=value['checkpoint_sha256']))
    result = dict(status='COMPLETE_EXPLORATORY_FUTURE_SPAN_FREQUENCY_DIAGNOSTIC',
        frequency_reference=baseline, records=rows, input_audits=inputs,
        main_report_sha256=file_sha256(main_root/'report.json'), registration_sha256=file_sha256(plan_file),
        wall_seconds_including_wait=time.time()-START, assigned_chips=0, model_forwards=0,
        optimizer_updates=0, test_accessed=False, terminal_alpha_spent=0, terminal_success=False,
        scope=plan['scope'], interpretation='Auxiliary-loss learning and NTP transfer are separate. The constant-frequency predictor measures an inexpensive alternative for this auxiliary target. It does not causally partition the network gain,identify the learned representation,or rescue a failed primary objective interaction.')
    write_state(out/'report.json', result);write_state(PUBLIC/'report.json', result)
    write_state(PUBLIC/'status.json', {'status':result['status'], 'assigned_chips':0})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        parser = argparse.ArgumentParser(add_help=False);parser.add_argument('--root', type=Path)
        args, _ = parser.parse_known_args()
        if args.root and (args.root/IDENTITY/'plan.json').exists() and not (args.root/IDENTITY/'report.json').exists():
            stopped = dict(status='STOPPED_EXPLORATORY_FREQUENCY_DIAGNOSTIC', error=str(error),
                wall_seconds_including_wait=time.time()-START, assigned_chips=0, model_forwards=0,
                optimizer_updates=0, test_accessed=False, terminal_success=False)
            write_state(args.root/IDENTITY/'report.json', stopped);write_state(PUBLIC/'report.json', stopped)
            write_state(PUBLIC/'status.json', {'status':stopped['status'], 'assigned_chips':0})
        raise
