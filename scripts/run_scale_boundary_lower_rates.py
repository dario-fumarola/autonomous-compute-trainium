"""One prospectively bounded symmetric lower-rate follow-up; never main."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY = 'scale-boundary-lower-rates-v1'
PARENT = 'scale-boundary-rate-completion-v2'
SCOPE = 'scale-boundary-lower-rate-scope-v1'
DOCUMENT = 'docs/SCALE_BOUNDARY_LOWER_RATE_CHECK_20261027.md'
PUBLIC = Path('results/scale_boundary_lower_rates_20261027')
GRID = [.002, .004, .008, .016, .032, .064]
COMPLETION = 'scale-boundary-last-original-rates-v1'
ORIGINAL_CERTIFICATE = 'scale-original-rate-closure-v1'
FINAL_RECOVERY = 'scale-final-lower-rate-recovery-v1'


def lower_selection_files(root):
    """Accept only the explicitly bound final opportunity after an inode stop."""
    original_plan=root/IDENTITY/'plan.json';original_report=root/IDENTITY/'report.json'
    stopped=json.loads(original_report.read_text())
    if stopped['status']=='COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION':
        return original_plan,original_report
    plan_file=root/FINAL_RECOVERY/'plan.json';report_file=root/FINAL_RECOVERY/'report.json'
    if (stopped['status']!='STOPPED_BOUNDARY_LOWER_RATES'
            or stopped.get('error')!='Filesystem inode reserve failed: '+str(root)
            or stopped['plan_sha256']!=file_sha256(original_plan)
            or stopped['records'].get('ordinary8-lower-lr1')!={}):
        raise ValueError('Exact stopped final opportunity required;no result substitution')
    plan=json.loads(plan_file.read_text());report=json.loads(report_file.read_text())
    if (plan['recovery_parent_plan_sha256']!=file_sha256(original_plan)
            or plan['selection_parent_report_sha256']!=file_sha256(original_report)
            or plan['case_order']!=['ordinary8-lower-lr1']
            or report['plan_sha256']!=file_sha256(plan_file)
            or report['status']!='COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION'):
        raise ValueError('Complete exact single-opportunity recovery required')
    for key,record in stopped['records'].items():
        if key!='ordinary8-lower-lr1' and report['records'].get(key)!=record:
            raise ValueError('Recovery changed an already completed opportunity')
    return plan_file,report_file


def original_selection_source(root):
    """An explicit completion certificate never overwrites its stopped parent."""
    plan_file=root/PARENT/'plan.json';report_file=root/PARENT/'report.json'
    report=json.loads(report_file.read_text())
    if report['status']=='COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION':
        if report['plan_sha256']!=file_sha256(plan_file):raise AssertionError('Original selection changed')
        return plan_file,report_file
    completion_plan=root/COMPLETION/'plan.json';completion_report=root/COMPLETION/'report.json'
    plan=json.loads(completion_plan.read_text());completed=json.loads(completion_report.read_text())
    if completed['status']!='COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION':
        certificate_plan=root/ORIGINAL_CERTIFICATE/'plan.json'
        certificate_report=root/ORIGINAL_CERTIFICATE/'report.json'
        certificate=json.loads(certificate_plan.read_text())
        if (certificate['completed_last_stage_plan_sha256']!=file_sha256(completion_plan)
                or certificate['completed_last_stage_report_sha256']!=file_sha256(completion_report)):
            raise AssertionError('Explicit original-grid certificate source changed')
        completion_plan,completion_report=certificate_plan,certificate_report
        plan=certificate;completed=json.loads(completion_report.read_text())
    if (completed['status']!='COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION'
            or completed['plan_sha256']!=file_sha256(completion_plan)
            or plan['original_stopped_parent_plan_sha256']!=file_sha256(plan_file)
            or plan['parent_report_sha256']!=file_sha256(report_file)
            or select(completed['records'],plan['selection_rates'],expected=4)!=completed['selected']):
        raise ValueError('Explicit bound completion of every original opportunity required')
    return completion_plan,completion_report


def needs_lower_check(selection):
    return any(selection[role]['rate'] == .008 for role in ['fixed_two', 'ordinary8'])


def selection_source(root):
    """Never silently fall back when the registered lower-edge trigger fires."""
    plan_file,report_file=original_selection_source(root)
    report = json.loads(report_file.read_text())
    if (report['status'] not in ['COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION','COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION']
            or report['plan_sha256'] != file_sha256(plan_file)):
        raise ValueError('Complete unchanged original rate selection required')
    scope_file = root / SCOPE / 'scope.json'
    if scope_file.exists():
        scope = json.loads(scope_file.read_text())
        if scope['parent_plan_sha256'] != file_sha256(root/PARENT/'plan.json'):
            raise AssertionError('Prospective lower-rate parent changed')
        if needs_lower_check(report['selected']):
            plan_file,lower_file=lower_selection_files(root)
            lower = json.loads(lower_file.read_text())
            plan = json.loads(plan_file.read_text())
            if (lower['status'] != 'COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION'
                    or lower['plan_sha256'] != file_sha256(plan_file)
                    or plan['parent_report_sha256'] != file_sha256(report_file)
                    or plan['lower_rate_scope_sha256'] != file_sha256(scope_file)
                    or plan['selection_rate_grid'] != GRID
                    or select(lower['records'], plan['selection_rates']) != lower['selected']):
                raise ValueError('Complete bound six-opportunity selection required')
            report_file = lower_file
    return plan_file, report_file


def select(records, rates, expected=6):
    selected = {}
    for role, prefix in [('fixed_two', 'fixed-two'), ('ordinary8', 'ordinary8')]:
        keys = [k for k in rates if k.startswith(prefix)]
        if len(keys) != expected or any(k not in records or ('score' not in records[k]
                and not records[k].get('declared_numerically_ineligible')) for k in keys):
            raise ValueError('All six fixed opportunities per role must complete')
        eligible = [k for k in keys if not records[k].get('declared_numerically_ineligible')
                    and records[k]['score']['metrics']['weighted_nll'] < 4.]
        if not eligible:
            raise ValueError('No healthy independent rate for ' + role)
        key = min(eligible, key=lambda k: (records[k]['score']['metrics']['weighted_nll'], rates[k]))
        boundaries=[min(rates[k] for k in keys),max(rates[k] for k in keys)]
        selected[role] = dict(case=key, rate=rates[key],
                              score=records[key]['score']['metrics']['weighted_nll'],
                              boundary_optimum=rates[key] in boundaries)
    return selected


def freeze(root):
    out = root / SCOPE
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every prospective scope')
    parent = root / PARENT / 'plan.json'
    if (parent.parent / 'report.json').exists():
        raise ValueError('Freeze before the original rate selection finishes')
    amendment = root / 'research-clock-accounting-amendment-20261027-v1/plan.json'
    value = dict(identity=SCOPE, registered_unix=time.time(),
                 document_sha256=file_sha256(DOCUMENT),
                 parent_plan_sha256=file_sha256(parent),
                 clock_amendment_sha256=file_sha256(amendment),
                 additional_rates=[.004, .002], conditional=True,
                 scientific_main_fit_admitted=False, scope=Path(DOCUMENT).read_text())
    out.mkdir(); PUBLIC.mkdir()
    write_state(out / 'scope.json', value); write_state(PUBLIC / 'scope.json', value)


def prepare(root):
    if managed_running():
        raise RuntimeError('Complete prior Native closure required')
    folder = root / IDENTITY
    if folder.exists():
        raise FileExistsError('Preserve every lower-rate attempt')
    scope_file = root / SCOPE / 'scope.json'
    scope = json.loads(scope_file.read_text())
    parent_file,result_file=original_selection_source(root)
    parent = json.loads(parent_file.read_text()); result = json.loads(result_file.read_text())
    if (scope['document_sha256'] != file_sha256(DOCUMENT)
            or scope['parent_plan_sha256'] != file_sha256(root/PARENT/'plan.json')
            or result['plan_sha256'] != file_sha256(parent_file)
            or result['status'] not in ['COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION','COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION']):
        raise ValueError('Complete unchanged parent and prospective scope required')
    if not needs_lower_check(result['selected']):
        folder.mkdir()
        value = dict(status='NOT_NEEDED_INTERIOR_OR_UPPER_ORIGINAL_SELECTION',
                     parent_report_sha256=file_sha256(result_file),
                     scope_sha256=file_sha256(scope_file), optimizer_updates=0)
        write_state(folder / 'report.json', value); write_state(PUBLIC / 'report.json', value)
        return False
    amendment = root / 'research-clock-accounting-amendment-20261027-v1/plan.json'
    clock = json.loads(amendment.read_text())
    if (file_sha256(amendment) != scope['clock_amendment_sha256']
            or not clock['overall_clock_admission_waived']
            or not clock['scientific_endpoints_unchanged']):
        raise ValueError('Explicit owner clock amendment required')
    if shutil.disk_usage(root).free < 53 * 1024**3:
        raise RuntimeError('Complete18GiB artifacts/failure peak plus35GiB reserve required')
    if shutil.disk_usage(root.parent / 'recurrence').free < (35 + 7) * 1024**3:
        raise RuntimeError('All four retained predictions plus reserve required')
    # Bind all exact original numerical proofs and the unchanged learned code.
    for name, sha in parent['numerical_gates'].items():
        if file_sha256(root / name / 'completion-verification.json') != sha:
            raise AssertionError('Exact numerical proof changed')
    for name, sha in parent['source_sha256'].items():
        if (name.startswith('compute_trainium/') or name in [
                'scripts/train_scale_registered.py', 'scripts/train_scale_capture_invalid_gradient.py',
                'scripts/audit_scale_registered.py', 'scripts/score_scale_registered.py']):
            if file_sha256(name) != sha:
                raise AssertionError('Qualified mathematical/learning source changed: ' + name)
    original = json.loads((root / 'scale-medium-tune-65536-v1/plan.json').read_text())
    cases = {}; phases = {}; names = []; limits = {}; order = []
    rates = {key: original['cases'][key]['peak_lr'] for key in original['case_order']}
    for index, rate in enumerate([.004, .002]):
        for prefix in ['fixed-two', 'ordinary8']:
            key = prefix + '-lower-lr' + str(index); order.append(key); rates[key] = rate
            case = dict(original['cases'][prefix + '-lr0'])
            if file_sha256(root / case['initializer']) != case['initializer_sha256']:
                raise AssertionError('Original untouched C0 changed')
            artifact = IDENTITY + '/artifacts/' + key
            case.update(peak_lr=rate, artifact_directory=artifact + '/warm', replay_directory=artifact + '/replay',
                        audit_directory=IDENTITY + '/audits/' + key, score_directory=IDENTITY + '/scores/' + key,
                        retained_model='../recurrence/' + IDENTITY + '/retained/' + key + '.pt',
                        cleanup_receipt=IDENTITY + '/cleanup/' + key + '.json')
            cases[key] = case
            cases[key + '-replay'] = {**case, 'artifact_directory': case['replay_directory'],
                                     'resume': dict(directory=case['artifact_directory'], update=299),
                                     'save_updates': [300], 'capture_before_updates': [], 'audit_first_update': False}
            for suffix, ck, seconds in [('warm', key, 1500), ('replay', key + '-replay', 600)]:
                name = IDENTITY + '-' + key + '-' + suffix
                names.append(name); phases[name] = dict(case=ck); limits[name] = seconds
    sources = {str(f): file_sha256(f) for d in ['compute_trainium', 'scripts', 'configs', 'third_party']
               for f in Path(d).rglob('*.py') if f.is_file()}
    for name in ['scripts/run_native_isolated.sh', DOCUMENT]:
        sources[name] = file_sha256(name)
    worker, host = split_snapshot_sources(sources); start = time.time()
    plan = {**parent, 'identity': IDENTITY, 'registered_unix': start,
            'source_sha256': sources, 'worker_source_sha256': worker, 'host_only_source_sha256': host,
            'seconds_limit': 6 * 3600, 'absolute_deadline_unix': start + 6 * 3600,
            'clock_admission_waiver': True, 'user_clock_amendment_sha256': file_sha256(amendment),
            'lower_rate_scope_sha256': file_sha256(scope_file), 'parent_plan_sha256': file_sha256(parent_file),
            'selection_parent_report':str(result_file.relative_to(root)),
            'parent_report_sha256': file_sha256(result_file), 'cases': cases, 'case_order': order,
            'selection_rates': rates, 'selection_rate_grid': GRID, 'phases': phases,
            'selection_expected_per_role':6,
            'complete_selection_status':'COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION',
            'stopped_selection_status':'STOPPED_BOUNDARY_LOWER_RATES',
            'root_comm_ports': reserve_ports(root, names), 'native_phase_seconds_limits': limits,
            'case_native_phases': {k: [IDENTITY + '-' + k + '-' + s for s in ['warm', 'replay']] for k in order},
            'allowed_tune_artifact_roots': [IDENTITY + '/artifacts'],
            'actual_new_tune_update_allowance': 4 * 301, 'scope': Path(DOCUMENT).read_text(),
            'scientific_main_fit_admitted': False}
    folder.mkdir()
    write_state(folder / 'plan.json', plan); write_state(PUBLIC / 'plan.json', plan)
    return True


def run(root):
    folder = root / IDENTITY
    execution = AllChipExecution(root, folder / 'plan.json', folder / 'execution')
    parent_file = root / execution.plan.get('selection_parent_report',PARENT+'/report.json')
    if file_sha256(parent_file) != execution.plan.get('selection_parent_report_sha256',execution.plan['parent_report_sha256']):
        raise AssertionError('Original complete selection changed')
    records = dict(json.loads(parent_file.read_text())['records'])
    records.update(execution.plan.get('declared_ineligible_records',{}))
    result = dict(status=execution.plan.get('stopped_selection_status','STOPPED_BOUNDARY_LOWER_RATES'), records=records, scientific_main_fit_admitted=False)
    def stop(number, frame):
        raise KeyboardInterrupt('Bounded lower-rate follow-up stopped')
    handlers = {s: signal.signal(s, stop) for s in [signal.SIGTERM, signal.SIGINT]}
    try:
        for key in execution.plan['case_order']:
            case = execution.plan['cases'][key]; record = {}; records[key] = record
            for suffix, ck in [('warm', key), ('replay', key + '-replay')]:
                name = IDENTITY + '-' + key + '-' + suffix
                options = ['--root', '/work/instance-nvme', '--plan', '/work/instance-nvme/' + IDENTITY + '/plan.json',
                           '--plan-sha', execution.sha, '--case', ck,
                           '--comm-port', str(execution.plan['root_comm_ports'][name])]
                record[suffix] = execution.native(name, 'scripts.train_scale_capture_invalid_gradient', options,
                                                  'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE')
                write_state(folder / 'progress.json', result)
            record['audit'] = execution.cpu(key + '-audit', execution.plan.get('audit_entry_module','scripts.audit_scale_registered'),
                ['--root', str(root), '--plan', str(folder / 'plan.json'), '--case', key,
                 '--output', str(root / case['audit_directory'])], root / case['audit_directory'] / 'report.json',
                'PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
            record['score'] = execution.cpu(key + '-score', 'scripts.score_scale_registered',
                ['--root', str(root), '--plan', str(folder / 'plan.json'), '--case', key,
                 '--output', str(root / case['score_directory'])], root / case['score_directory'] / 'report.json',
                'COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
            cleanup = subprocess.run(['sudo', '-n', 'env', 'USE_TORCH_XLA=0', 'PYTHONDONTWRITEBYTECODE=1',
                'PYTHONPATH=' + str(Path('.').resolve()), os.sys.executable, '-B', '-m',
                'scripts.close_scale_tune_diagnostics', '--root', str(root), '--plan', str(folder / 'plan.json'),
                '--case', key], capture_output=True, text=True, timeout=300)
            if cleanup.returncode:
                raise RuntimeError('Successful-only closure failed: ' + cleanup.stderr[-1000:])
            record['cleanup_receipt_sha256'] = file_sha256(root / case['cleanup_receipt'])
            write_state(folder / 'progress.json', result)
        result.update(status=execution.plan.get('complete_selection_status','COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION'),
                      selected=select(records, execution.plan['selection_rates'],execution.plan.get('selection_expected_per_role',6)))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        execution.close(folder / 'execution-costs.json'); result['plan_sha256'] = execution.sha
        write_state(folder / 'report.json', result); write_state(PUBLIC / 'report.json', result)
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--freeze', action='store_true'); p.add_argument('--run', action='store_true')
    a = p.parse_args()
    freeze(a.root) if a.freeze else (run(a.root) if a.run else prepare(a.root))
