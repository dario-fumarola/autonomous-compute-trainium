"""Conditional within-family work controls,with gated counted continuations."""
import argparse
import concurrent.futures as cf
import copy
import json
import signal
import time
from pathlib import Path

import torch

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_gradient_point import point_tables
from compute_trainium.future_span_transport import transport_stable
from compute_trainium.future_span_work_control import (
    IDENTITY, QUEUE_IDENTITY, admission_evidence, make_cases, measured_control_budget,
    validate_control_score,
)
from compute_trainium.metrics import DocumentScore, paired_group_interval
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint, file_sha256
from scripts.qualify_native import digest

PUBLIC = Path('results/future_span_work_controls_20261025')


def source_integrity(sources):
    if any(file_sha256(p) != sha for p, sha in sources.items()):
        raise AssertionError('Frozen work-control source changed')


def prepare_queue(root):
    out = root / QUEUE_IDENTITY
    file = out / 'plan.json'
    if file.exists():
        plan = json.loads(file.read_text())
        source_integrity(plan['source_sha256'])
        return plan
    main_file = root / 'future-span-main-v2/plan.json'
    main = json.loads(main_file.read_text())
    sources = dict(main['source_sha256'])
    inputs = {}
    for name, leaf in [('future-span-compute-readiness-v1', 'report.json'),
                       ('future-span-transport-qualification-v1', 'report.json'),
                       ('ntp-secondary-discovery-v1', 'plan.json')]:
        path = root / name / leaf
        value = json.loads(path.read_text())
        required = {'future-span-compute-readiness-v1': 'PASS_PROSPECTIVE_WORK_MATCHED_CONTINUATION_GEOMETRY',
                    'future-span-transport-qualification-v1': 'PASS_CPU_FUTURE_SPAN_STATE_TRANSPORT'}
        if name in required:
            binding = value.get('main_plan_sha256', value.get('source_plan_sha256'))
            if value.get('status') != required[name] or binding != file_sha256(main_file):
                raise AssertionError('Completed original geometry/transport prerequisite required')
        inputs[str(path.relative_to(root))] = file_sha256(path)
        for p, sha in value.get('analysis_source_sha256', value.get('source_sha256', {})).items():
            if p in sources and sources[p] != sha:
                raise AssertionError('Prerequisite scientific source conflicts')
            sources[p] = sha
    for p in ['compute_trainium/future_span_work_control.py',
              'scripts/run_future_span_work_controls.py', 'tests/test_future_span_work_control.py',
              'docs/FUTURE_SPAN_WORK_CONTROLS_20261025.md']:
        sources[p] = file_sha256(p)
    source_integrity(sources)
    hardware_file = root / 'future-span-math-qualification-v2/report.json'
    if file_sha256(hardware_file) != main['launch_receipts']['future-span-math-qualification-v2']['report_sha256']:
        raise AssertionError('Original exact-geometry measurement changed')
    budget = measured_control_budget(json.loads(hardware_file.read_text()))
    if not budget['fits_registered_cap']:
        raise ValueError('Measured conservative controls exceed the prospective3h cap')
    execution = json.loads((root / 'future-span-main-v2/execution/execution-start.json').read_text())
    plan = {'identity': QUEUE_IDENTITY, 'registered_unix': time.time(),
        'main_plan_sha256': file_sha256(main_file), 'source_sha256': sources,
        'prerequisite_input_sha256': inputs, 'hardware_report_sha256': file_sha256(hardware_file),
        'wait_deadline_unix': execution['deadline_unix'], 'measured_budget': budget,
        'candidate': 'recurrent-ntp at its fixed two-pass cooled1B exit',
        'conditions': 'Complete unchanged MAIN and registered NTP secondary;both seeds clear every original NTP point gate;no other native owners;budget and all source/state checks pass.',
        'registration_disclosure': 'Prepared after complete seed5231 discovery outcomes,while seed7283 runs. No matched-work outcomes exist. Geometry was independently determined before MAIN quality outcomes. This is a conditional development control,not confirmation.',
        'assigned_chips': 0, 'test_accessed': False, 'terminal_alpha_spent': 0,
        'scope': 'Four within-family controls:one-NTP/one-future in both paired seeds,each own complete work-matched cooldown. Stronger ordinary/cheap baselines and inference chip-time costs remain owed. No launch if discovery,prerequisite,budget or source gates fail.'}
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(file, plan)
    write_state(PUBLIC / 'queue-plan.json', plan)
    return plan


def wait_and_register(root, queue):
    main_file = root / 'future-span-main-v2/plan.json'
    main = json.loads(main_file.read_text())
    while True:
        source_integrity(queue['source_sha256'])
        if time.time() > queue['wait_deadline_unix']:
            raise TimeoutError('Original parent admission window expired')
        if (file_sha256(main_file) != queue['main_plan_sha256']
                or any(file_sha256(root / p) != sha for p, sha in queue['prerequisite_input_sha256'].items())):
            raise AssertionError('Original prerequisite changed')
        parent_file = root / 'future-span-main-v2/report.json'
        secondary_file = root / 'ntp-secondary-discovery-v1/report.json'
        for p in [parent_file, secondary_file]:
            if p.exists() and json.loads(p.read_text())['status'].startswith('STOPPED'):
                raise RuntimeError('Parent stopped;do not admit work controls')
        if parent_file.exists() and secondary_file.exists() and not managed_running():
            parent = json.loads(parent_file.read_text())
            secondary = json.loads(secondary_file.read_text())
            if (parent['plan_sha256'] != queue['main_plan_sha256']
                    or secondary['parent_report_sha256'] != file_sha256(parent_file)
                    or secondary['parent_plan_sha256'] != queue['main_plan_sha256']
                    or secondary['registration_sha256'] != file_sha256(root / 'ntp-secondary-discovery-v1/plan.json')):
                raise AssertionError('Secondary and original MAIN identities differ')
            evidence = admission_evidence(parent, main, secondary)
            if not evidence['eligible']:
                result = {'status': 'NOT_ADMITTED_NTP_DISCOVERY_GATES_FAILED',
                    'evidence': evidence, 'assigned_chips': 0, 'test_accessed': False}
                write_state(root / QUEUE_IDENTITY / 'report.json', result)
                write_state(PUBLIC / 'queue-report.json', result)
                return None
            break
        write_state(PUBLIC / 'status.json', {'status': 'WAITING_FOR_COMPLETE_MAIN_AND_BOTH_NTP_DISCOVERY_PAIRS',
                                          'assigned_chips': 0})
        time.sleep(10)
    out = root / IDENTITY
    if out.exists():
        raise FileExistsError('Existing admitted controls require explicit resume')
    geometry_file = root / 'future-span-compute-readiness-v1/report.json'
    geometry = json.loads(geometry_file.read_text())['one_pass_control_branch']
    aux = main['auxiliary_data']
    anchor_file = root / aux['directory'] / aux['prefix_file']
    if file_sha256(anchor_file) != aux['prefix_sha256']:
        raise AssertionError('Original source-anchor prefix changed')
    import numpy as np
    cases = make_cases(main, geometry, np.load(anchor_file, mmap_mode='r'))
    parents = {}
    for key, case in cases.items():
        old = case['source_parent_case_id']
        identity = 'future-span-main-v2-' + old + '-first'
        folder = root / 'runs' / identity
        report_file = folder / 'report.json'
        closure_file = root / 'snapshots' / identity / 'closure.json'
        report = json.loads(report_file.read_text())
        row = next(r for r in report['checkpoint_records'] if r['update'] == geometry['origin_update'])
        checkpoint = folder / f'checkpoint-u{geometry["origin_update"]}.pt'
        if (report['plan_sha256'] != queue['main_plan_sha256'] or report['case_id'] != old
                or report['qualification_only'] or not row['all_complete_replica_states_equal']
                or file_sha256(checkpoint) != row['checkpoint_sha256']
                or json.loads(closure_file.read_text())['exit_code'] != 0):
            raise ValueError('Exact immutable original stable parent required')
        parents[key] = {'source_case_id': old, 'file': str(checkpoint.relative_to(root)),
            'sha256': row['checkpoint_sha256'], 'full_state_digest': row['full_state_digest'],
            'native_report_sha256': file_sha256(report_file), 'native_closure_sha256': file_sha256(closure_file)}
    identities = [IDENTITY + '-' + k + '-' + phase for k in cases
                  for phase in ['preflight', 'preflight-gradient', 'finish', 'final-gradient']]
    plan = copy.deepcopy(main)
    plan.update(identity=IDENTITY, stage='work_matched_continuations', registered_unix=time.time(),
        source_sha256=queue['source_sha256'], worker_source_sha256=split_snapshot_sources(queue['source_sha256'])[0],
        cases=cases, source_parents=parents, root_comm_ports=reserve_ports(root, identities),
        seconds_limit=queue['measured_budget']['execution_seconds_cap'], phase_seconds_limit=3600,
        cpu_phase_seconds_limit=1200, cpu_workers=4, measured_budget=queue['measured_budget'],
        source_main_plan_sha256=queue['main_plan_sha256'], source_main_report_sha256=file_sha256(parent_file),
        source_secondary_report_sha256=file_sha256(secondary_file),
        queue_plan_sha256=file_sha256(root / QUEUE_IDENTITY / 'plan.json'),
        control_geometry=geometry, admission_evidence=evidence,
        preflight_policy='Two actual scientific continuation updates per case,counted in the final endpoint. Exact model/optimizer/RNG loading and complete512-gradient audit in all four cases precede every long continuation. No qualification initializer or nonlaunchable qualification copy is used.',
        scope='Conditional development within-family work controls. Fixed two-pass NTP candidate from complete MAIN,against both independently tuned one-pass objectives at complete matched nominal matrix work. Every endpoint owns its cooldown. Inherited prefix is charged;incremental campaign cost is separate. Not strongest-ordinary,wall-time-matched or terminal confirmation.')
    out.mkdir()
    write_state(out / 'plan.json', plan)
    write_state(PUBLIC / 'plan.json', plan)
    return plan


def run(root, plan, *, resume=False):
    out = root / IDENTITY
    plan_file = out / 'plan.json'
    if (out / 'report.json').exists():
        raise FileExistsError('Preserve terminal work controls')
    source_file = root / 'future-span-main-v2/plan.json'
    source = json.loads(source_file.read_text())
    if file_sha256(source_file) != plan['source_main_plan_sha256']:
        raise AssertionError('Original model history changed')
    execution = Execution(root, plan_file, out / 'execution', resume=resume)
    sha = execution.sha
    cpu = cf.ThreadPoolExecutor(plan['cpu_workers'])
    fits, gradients, scores, transports = [], [], [], []
    result = {'status': 'STOPPED_FUTURE_SPAN_WORK_CONTROLS'}
    def interrupted(signum, frame):
        execution.failed.set()
        raise KeyboardInterrupt('Work control interrupted by signal ' + str(signum))
    handlers = {s: signal.signal(s, interrupted) for s in [signal.SIGINT, signal.SIGTERM]}
    def checkpoint(report):
        return next(r for r in report['checkpoint_records'] if r['update'] == report['update'])
    def transport(key):
        execution.check()
        case = plan['cases'][key]
        parent = plan['source_parents'][key]
        target = out / 'transport' / (key + '.pt')
        receipt = target.with_suffix('.json')
        original = root / parent['file']
        if file_sha256(original) != parent['sha256']:
            raise AssertionError('Immutable original parent changed')
        if receipt.exists():
            proof = json.loads(receipt.read_text())
            if (proof['destination_plan_sha256'] != sha or proof['source_checkpoint_sha256'] != parent['sha256']
                    or file_sha256(target) != proof['derived_checkpoint_sha256']):
                raise AssertionError('Previously transported parent differs')
            return target, proof
        if target.exists():
            raise RuntimeError('Partial state transport needs explicit recovery')
        begin = time.time()
        counts, anchors, _, work = point_tables(root, plan_file, plan, case)
        cursor = case['parent']['update']*512
        counters = dict(expected_targets=int((counts[cursor]-counts[0]).sum()),
            expected_domains=list(map(int, counts[cursor]-counts[0])), expected_anchors=int(anchors[cursor]),
            expected_matrix_ops=sum(map(int, work[:case['parent']['update']])))
        saved = torch.load(original, map_location='cpu', mmap=True, weights_only=False)
        new, proof = transport_stable(saved, source_plan=source, source_plan_sha=plan['source_main_plan_sha256'],
            source_case_id=parent['source_case_id'], source_checkpoint_sha=parent['sha256'],
            destination_plan=plan, destination_plan_sha=sha, destination_case_id=key, expected_counters=counters)
        if proof['complete_model_optimizer_digest'] != parent['full_state_digest']:
            raise AssertionError('Original native state digest differs')
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_checkpoint(target, new)
        restored = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        if (digest({'model': dict(sorted(restored['model'].items())), 'optimizer': restored['optimizer']})
                != parent['full_state_digest'] or digest(restored['cpu_rng_by_rank']) != proof['cpu_rng_digest']
                or restored['signature'] != new['signature'] or file_sha256(original) != parent['sha256']):
            raise AssertionError('Transport changed tensors,RNG or original state')
        proof.update(derived_checkpoint_sha256=file_sha256(target), wall_seconds=time.time()-begin,
                     status='PASS_IMMUTABLE_REGISTERED_PARENT_TRANSPORT', assigned_chips=0)
        write_state(receipt, proof)
        return target, proof
    def fit(key, phase, end, file, parent_sha, expected_digest):
        case = plan['cases'][key]
        name = IDENTITY + '-' + key + '-' + phase
        options = ['--root', '/work/instance-nvme', '--plan', '/work/instance-nvme/' + IDENTITY + '/plan.json',
            '--plan-sha', sha, '--case', key, '--segment-end', str(end), '--comm-port', str(plan['root_comm_ports'][name]),
            '--resume', '/work/instance-nvme/' + str(file.relative_to(root)), '--resume-sha', parent_sha]
        report = execution.native(name, 'scripts.train_future_span_primitives', options,
                                  'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        expected_updates = end-(case['parent']['update'] if phase == 'preflight' else case['preflight_end_update'])
        if (report['case_id'] != key or report['plan_sha256'] != sha or report['update'] != end
                or report['qualification_only'] or not report['complete_state_on_load_exact']
                or not report['cpu_rng_on_load_exact'] or report['recovered_state_digest'] != expected_digest
                or report['research_optimizer_updates'] != expected_updates
                or report['actual_optimizer_calls_per_replica'] != expected_updates):
            raise AssertionError('Actual counted continuation/recovery differs')
        row = checkpoint(report)
        if file_sha256(root / 'runs' / name / f'checkpoint-u{end}.pt') != row['checkpoint_sha256'] or not row['all_complete_replica_states_equal']:
            raise AssertionError('Complete scientific checkpoint/replicas differ')
        if end == case['total_updates'] and (report['real_targets'] != case['end_targets']
                or report['estimated_dense_matrix_ops'] != case['training_dense_matrix_ops'] or report['valid_anchors'] != case['end_anchors']):
            raise AssertionError('Final complete work/source/anchor budget differs')
        with execution.lock:
            fits.append({'identity': name, **{k:v for k,v in report.items() if k not in ['progress', 'first_global_probe']}})
            write_state(PUBLIC / 'training-progress.json', {'records': fits})
        return name, report
    def gradient(key, capture, phase):
        name, report = capture
        row = checkpoint(report)
        update = report['update']
        identity = IDENTITY + '-' + key + '-' + phase
        file = root / 'runs' / name / f'checkpoint-u{update}.pt'
        common = ['--plan-sha', sha, '--case', key, '--checkpoint-sha', row['checkpoint_sha256']]
        native = execution.native(identity, 'scripts.capture_future_span_primitives',
            ['--root', '/work/instance-nvme', '--plan', '/work/instance-nvme/' + IDENTITY + '/plan.json', *common,
             '--checkpoint', '/work/instance-nvme/' + str(file.relative_to(root)),
             '--comm-port', str(plan['root_comm_ports'][identity])], 'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        if native['checkpoint_sha256'] != row['checkpoint_sha256'] or native['plan_sha256'] != sha:
            raise AssertionError('Native continuation gradient point differs')
        target = out / 'gradients' / f'{key}-u{update}'
        future = cpu.submit(execution.cpu, f'{key}-u{update}-gradient-audit', 'scripts.audit_future_span_gradient',
            ['--root', str(root), '--plan', str(plan_file), *common, '--checkpoint', str(file),
             '--capture', str(root / 'runs' / identity), '--output', str(target),
             '--upstream', str(root / 'upstream-parcae-69284c1'), '--seconds-limit', '1200'],
            target / 'report.json', 'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        return identity, future
    def native_job(key, phase, parents):
        try:
            case = plan['cases'][key]
            if phase == 'preflight':
                file, proof = transport(key)
                with execution.lock: transports.append({'case_id': key, **proof})
                capture = fit(key, phase, case['preflight_end_update'], file,
                              proof['derived_checkpoint_sha256'], proof['complete_model_optimizer_digest'])
                point, audit = gradient(key, capture, 'preflight-gradient')
            else:
                name, previous = parents[key]
                row = checkpoint(previous)
                capture = fit(key, phase, case['total_updates'],
                    root / 'runs' / name / f'checkpoint-u{previous["update"]}.pt', row['checkpoint_sha256'], row['full_state_digest'])
                point, audit = gradient(key, capture, 'final-gradient')
            return capture, point, audit
        except BaseException:
            execution.failed.set()
            raise
    def accept_gradient(key, capture, point, audit):
        checked = audit.result(timeout=max(1, execution.deadline-time.time()))
        if (checked['checkpoint_sha256'] != checkpoint(capture[1])['checkpoint_sha256']
                or checked['training_update'] != capture[1]['update']):
            execution.failed.set()
            raise AssertionError('Independent audit covers another endpoint')
        record = {'case_id': key, 'update': capture[1]['update'], 'native_identity': point,
            'native_report_sha256': file_sha256(root / 'runs' / point / 'report.json'), 'independent_cpu': checked}
        gradients.append(record)
        write_state(PUBLIC / 'gradient-progress.json', {'records': gradients})
    def score(key, capture):
        name, report = capture
        row = checkpoint(report)
        target = out / 'scores' / key
        checked = execution.cpu(key + '-score', 'scripts.evaluate_future_span_cpu',
            ['--root', str(root), '--plan', str(plan_file), '--plan-sha', sha, '--checkpoint',
             str(root / 'runs' / name / f'checkpoint-u{report["update"]}.pt'), '--checkpoint-sha', row['checkpoint_sha256'],
             '--case', key, '--cohort', 'curve', '--depth', '1', '--output', str(target),
             '--upstream', str(root / 'upstream-parcae-69284c1'), '--seconds-limit', '1200'],
            target / 'report.json', 'COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
        try:
            validate_control_score(checked, plan['cases'][key], key, row['checkpoint_sha256'], plan)
        except BaseException:
            execution.failed.set()
            raise
        with execution.lock:
            scores.append(checked)
            write_state(PUBLIC / 'scores-progress.json', {'records': scores})
        return checked
    try:
        if file_sha256(root / 'future-span-main-v2/report.json') != plan['source_main_report_sha256']:
            raise AssertionError('Original complete discovery result changed')
        parents = {}
        with cf.ThreadPoolExecutor(2) as native:
            jobs = {k: native.submit(native_job, k, 'preflight', None) for k in plan['cases']}
            for key, future in jobs.items():
                capture, point, audit = future.result()
                accept_gradient(key, capture, point, audit)
                parents[key] = capture
        if len(parents) != 4 or len(gradients) != 4:
            raise AssertionError('All four native preflight/restore/numerical gates precede long training')
        write_state(PUBLIC / 'preflight-pass.json', {'status': 'PASS_ALL_FOUR_COUNTED_CONTINUATION_PREFLIGHTS',
            'case_ids': list(parents), 'scientific_updates_per_case': 2, 'included_in_final_endpoint': True,
            'qualification_initializer_used': False})
        handles = []
        with cf.ThreadPoolExecutor(2) as native:
            jobs = {k: native.submit(native_job, k, 'finish', parents) for k in plan['cases']}
            for key, future in jobs.items():
                capture, point, audit = future.result()
                accept_gradient(key, capture, point, audit)
                handles.append(cpu.submit(score, key, capture))
        for future in handles:
            future.result(timeout=max(1, execution.deadline-time.time()))
        if ({r['case_id'] for r in scores} != set(plan['cases']) or len(scores) != 4
                or len(gradients) != 8
                or {(r['case_id'], r['update']) for r in gradients} != {
                    (k, u) for k,c in plan['cases'].items() for u in c['gradient_updates']}):
            raise AssertionError('All unique controls and both numerical points required')
        original = json.loads((root / 'future-span-main-v2/report.json').read_text())
        contrasts = {}
        for seed in [5231, 7283]:
            candidate = next(r for r in original['scores'] if r['seed'] == seed and r['cell'] == 'recurrent-ntp'
                             and r['depth'] == 2 and r['readout'] == 'cooled1b')
            file = root / 'future-span-main-v2/scores' / f'{candidate["case_id"]}-cooled1b-d2/document-scores.json'
            to_scores = lambda p: [DocumentScore(r['domain'], r['id'], r['group'], r['nll_sum'], r['real_targets'])
                                   for r in json.loads(p.read_text())['records']]
            target = to_scores(file)
            contrasts[seed] = {}
            for row in [r for r in scores if r['seed'] == seed]:
                control = to_scores(out / 'scores' / row['case_id'] / 'document-scores.json')
                contrasts[seed][row['cell']] = paired_group_interval(control, target, draws=2000, seed=91357+seed)
        result = {'status': 'COMPLETE_WITHIN_FAMILY_WORK_MATCHED_CONTROLS', 'scores': scores,
            'candidate_gains_over_longer_controls': contrasts,
            'all_screening_points_clear_useful_gain': all(v['gain_nats'] >= .01 for s in contrasts.values() for v in s.values()),
            'gradient_gates': gradients, 'transports': transports}
        execution.check()
    except BaseException as error:
        execution.failed.set()
        result.update(error=str(error), scores=scores, gradient_gates=gradients, training_records=fits, transports=transports)
        raise
    finally:
        execution.failed.set()
        cpu.shutdown(wait=True, cancel_futures=True)
        execution.close(PUBLIC / 'execution-costs.json')
        for signum, handler in handlers.items(): signal.signal(signum, handler)
        result.update(plan_sha256=sha, test_accessed=False, terminal_alpha_spent=0,
            terminal_success=False, scope=plan['scope'])
        write_state(out / 'report.json', result)
        write_state(PUBLIC / 'report.json', result)
        write_state(PUBLIC / 'status.json', {'status': result['status'], 'assigned_chips': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Work-control queue interrupted by signal ' + str(signum))
    for signum in [signal.SIGINT, signal.SIGTERM]: signal.signal(signum, interrupted)
    queue = prepare_queue(args.root)
    if args.prepare_only: return
    if args.resume:
        plan = json.loads((args.root / IDENTITY / 'plan.json').read_text())
        run(args.root, plan, resume=True)
        return
    plan = wait_and_register(args.root, queue)
    if plan is not None:
        run(args.root, plan)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--root', type=Path)
        args, _ = parser.parse_known_args()
        if args.root and (args.root / QUEUE_IDENTITY / 'plan.json').exists() and not (args.root / IDENTITY / 'plan.json').exists():
            result = {'status': 'STOPPED_BEFORE_WORK_CONTROL_ADMISSION', 'error': str(error), 'assigned_chips': 0,
                      'test_accessed': False, 'terminal_success': False}
            write_state(args.root / QUEUE_IDENTITY / 'report.json', result)
            write_state(PUBLIC / 'queue-report.json', result)
        raise
