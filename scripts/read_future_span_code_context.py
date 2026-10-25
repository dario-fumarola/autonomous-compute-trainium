"""Carry forward the proposed149-target exploratory diagnostic after full MAIN."""
import argparse
import gc
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA', '0')
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from compute_trainium.code_context_views import views
from compute_trainium.future_span_context import context_cells
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.parcae_candidate import ParcaeCandidateConfig, matmul_ledger
from compute_trainium.parcae_scientific_state import schedule_identity
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN

IDENTITY = 'future-span-code-context-v1'
PUBLIC = Path('results/future_span_code_context_20261025')
ATTEMPT_STARTED = time.time()
RUNTIME = {'forward_calls': 0, 'estimated_forward_matrix_ops': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.root / IDENTITY
    plan_file = out / 'plan.json'
    main_root = args.root / 'future-span-main-v2'
    parent_file = main_root / 'plan.json'
    parent_plan = json.loads(parent_file.read_text())
    selection = args.root / 'natural-code-context-v1'
    if not plan_file.exists():
        if (main_root / 'report.json').exists():
            raise ValueError('Register this exploratory carry-forward before MAIN is complete')
        if any((main_root / 'scores').glob('recurrent-*/report.json')):
            raise ValueError('Register before new MAIN recurrent quality outcomes')
        proposal = 'docs/FUTURE_SPAN_OBJECTIVE_PROPOSAL_20261025.md'
        if 'Carry forward the syntax-selected natural-code context diagnostic' not in Path(proposal).read_text():
            raise ValueError('The original proposal must require this diagnostic')
        selected = json.loads((selection / 'scoring-views-v3-report.json').read_text())
        qualified = json.loads((selection / 'intervention-qualification-v3.json').read_text())
        if (selected['selected_targets'] != 149
                or selected['panel_sha256'] != file_sha256(selection / 'scoring-panel-v3.json')
                or qualified['status'] != 'PASS_POSITION_NOISE_PRESERVING_CONTEXT_INTERVENTION'
                or qualified['panel_sha256'] != selected['panel_sha256']):
            raise ValueError('The unchanged qualified149-target panel is required')
        sources = dict(parent_plan['source_sha256'])
        for file, sha in qualified['source_sha256'].items():
            if file_sha256(file) != sha:
                raise AssertionError('The original position/noise intervention changed')
            sources[file] = sha
        for file in [__file__, 'compute_trainium/future_span_context.py',
                     'tests/test_future_span_context.py', 'compute_trainium/code_context_control.py',
                     'compute_trainium/code_context_views.py', proposal]:
            path = str(Path(file).resolve().relative_to(Path.cwd()))
            sources[path] = file_sha256(path)
        execution = json.loads((main_root / 'execution/execution-start.json').read_text())
        plan = {'identity': IDENTITY, 'registered_unix': time.time(),
            'main_plan_sha256': file_sha256(parent_file), 'source_sha256': sources,
            'frozen_inputs': {str((selection / name).relative_to(args.root)): file_sha256(selection / name)
                for name in ['plan.json', 'panel.json', 'scoring-panel-v3.json',
                             'scoring-views-v3-report.json', 'intervention-qualification-v3.json']},
            'panel_sha256': selected['panel_sha256'], 'upstream_commit': PIN,
            'original_proposal_commit': 'ed70c0c',
            'registration_disclosure': 'The carry-forward requirement and original selection/views predate MAIN. This execution is registered after the twelve TUNE outcomes and first seed5231 one-pass stable/cooled results;no new MAIN recurrent quality has been read. Prior small-core code-panel outcomes were consumed. This is exploratory,not a new preregistered confirmatory claim.',
            'seeds': [5231, 7283], 'cells': ['one-ntp', 'one-future', 'recurrent-ntp', 'recurrent-future'],
            'one_pass_depths': [1], 'recurrent_depths': [1, 2],
            'bootstrap_draws': 5000, 'bootstrap_seed': 1381,
            'wait_deadline_unix': execution['deadline_unix'], 'cpu_seconds_cap': 1800,
            'cpu_seconds_per_model': 240, 'assigned_chips': 0, 'optimizer_updates': 0,
            'test_accessed': False, 'terminal_alpha_spent': 0,
            'scope': 'All eight cooled MAIN models and exact endpoint gradient gates required. Unchanged149 authentic identifier targets,1008/128-prefix views,positions and prefix-only noise. Report within-model full benefit plus context interaction and independently trained one-pass controls in both objectives. Equal original distance-band weights. Repository intervals are conditional on each fitted seed. No new targets,teacher,synthetic corpus,primary-rule change or terminal claim.'}
        out.mkdir()
        PUBLIC.mkdir(parents=True)
        write_state(plan_file, plan)
        write_state(PUBLIC / 'plan.json', plan)
    else:
        plan = json.loads(plan_file.read_text())
    if args.prepare_only:
        return
    if (out / 'report.json').exists():
        raise FileExistsError('Preserve an existing terminal context readout')
    def guard():
        if (file_sha256(parent_file) != plan['main_plan_sha256']
                or any(file_sha256(f) != sha for f, sha in plan['source_sha256'].items())
                or any(file_sha256(args.root / f) != sha for f, sha in plan['frozen_inputs'].items())):
            raise AssertionError('Frozen MAIN,context source or original panel changed')
    while True:
        guard()
        file = main_root / 'report.json'
        if file.exists():
            parent = json.loads(file.read_text())
            if parent['status'] != 'COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT':
                raise RuntimeError('Do not select favorable partial/failed MAIN cases')
            break
        if time.time() > plan['wait_deadline_unix']:
            raise TimeoutError('Original MAIN window expired')
        write_state(PUBLIC / 'status.json', {'status': 'WAITING_FOR_ALL_MAIN_MODELS_AND_GRADIENT_GATES',
                                          'assigned_chips': 0})
        time.sleep(10)
    started = time.time()
    RUNTIME['compute_started_unix'] = started
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    upstream = args.root / 'upstream-parcae-69284c1'
    if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PIN
            or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Clean pinned independent upstream required')
    sys.path.insert(0, str(upstream))
    if (len(parent['independent_gradient_gates']) != 8
            or {g['case_id'] for g in parent['independent_gradient_gates']} != set(parent_plan['cases'])):
        raise ValueError('All eight unique MAIN numerical gates required')
    source = args.root / parent_plan['evaluation']['corpus']
    original = json.loads((selection / 'plan.json').read_text())
    if file_sha256(source) != original['source_sha256']:
        raise AssertionError('Original authentic-code source changed')
    docs = {d['id']: d for d in json.loads(source.read_text())}
    identifiers = json.loads((selection / 'scoring-panel-v3.json').read_text())['records']
    if len(identifiers) != 149:
        raise AssertionError('All original identifier targets required')
    config = ParcaeCandidateConfig(**parent_plan['model'])
    per_seed = {}
    receipts = []
    calls = matrix = 0
    def score_guard(begin):
        guard()
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if (time.time() - started > plan['cpu_seconds_cap'] or time.time() - begin > plan['cpu_seconds_per_model']
                or available < 544*1024**3 or min(shutil.disk_usage(out).free, shutil.disk_usage('/').free) < 35*1024**3):
            raise RuntimeError('Fixed CPU context-readout cap or resource reserve')
    for seed in plan['seeds']:
        cells = {}
        for cell in plan['cells']:
            begin = time.time()
            score_guard(begin)
            key, case = next((k, c) for k, c in parent_plan['cases'].items() if c['seed'] == seed and c['cell'] == cell)
            gate = next(g['independent_cpu'] for g in parent['independent_gradient_gates'] if g['case_id'] == key)
            checkpoint = args.root / 'runs' / ('future-span-main-v2-' + key + '-finish') / 'checkpoint-u1920.pt'
            if (gate['status'] != 'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                    or gate['training_update'] != case['total_updates']
                    or file_sha256(checkpoint) != gate['checkpoint_sha256']):
                raise ValueError('Exact completed checkpoint and numerical gate required')
            saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            if (saved['qualification_only'] is not False or saved['case_id'] != key or saved['update'] != case['total_updates']
                    or saved['signature'] != signature(plan['main_plan_sha256'], parent_plan, case)
                    or saved['schedule'] != schedule_identity(case)
                    or saved['cursor'] != saved['update']*512 or saved['real_targets'] != case['end_targets']):
                raise ValueError('Complete unchanged scientific endpoint required')
            full_digest = digest({'model': dict(sorted(saved['model'].items())), 'optimizer': saved['optimizer']})
            if full_digest != saved['model_optimizer_digest']:
                raise AssertionError('Complete endpoint readback differs')
            model = make_reference(config, True).eval()
            model.load_state_dict(prediction_state(saved['model']), strict=True)
            del saved
            before = digest(dict(sorted(model.state_dict().items())))
            current = None
            model.initialize_state = lambda _: current
            rows = []
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for i, point in enumerate(identifiers):
                    score_guard(begin)
                    document = docs[point['document_id']]
                    if hashlib.sha256(json.dumps(document['tokens']).encode()).hexdigest() != point['original_document_token_sha256']:
                        raise AssertionError('Original identifier tokens changed')
                    noise = initial_state(config, point['state_key'], case['state_seed'])
                    losses = {}
                    for label, view in views(document['tokens'], point).items():
                        tokens, targets = torch.tensor([view['inputs']]), torch.tensor(view['targets'])
                        positions = torch.tensor(view['positions'])
                        for depth in plan['one_pass_depths'] if case['arm'] == 'one' else plan['recurrent_depths']:
                            current = noise[:, view['state_start']:view['state_start'] + len(view['inputs'])].clone()
                            logits = model(tokens, position_ids=positions, num_steps_pair=torch.tensor([depth-1, 1]),
                                           return_logits=True)['logits'][0, view['prediction_positions']].double()
                            value = (logits.logsumexp(-1) - logits.gather(-1, targets[:, None]).squeeze(-1)).sum()
                            if not torch.isfinite(value):
                                raise FloatingPointError('Nonfinite authentic identifier NLL')
                            losses[label if case['arm'] == 'one' else label + str(depth)] = float(value)
                            calls += 1
                            matrix += matmul_ledger(config, len(view['inputs']), total_steps=depth, backward_cap=1)['forward_matmul_flops']
                            RUNTIME.update(forward_calls=calls, estimated_forward_matrix_ops=matrix)
                    rows.append({'id': point['id'], 'group': point['group'], 'band': point['band'],
                        'real_targets': point['target_end_token'] - point['target_start_token'], 'nll_sum': losses})
                    if (i+1) % 20 == 0:
                        write_state(PUBLIC / 'status.json', {'status': 'SCORING_UNCHANGED_NATURAL_CODE_CONTEXT',
                            'seed': seed, 'cell': cell, 'identifiers': i+1, 'assigned_chips': 0})
            if digest(dict(sorted(model.state_dict().items()))) != before:
                raise AssertionError('Code readout changed model weights')
            private = out / (key + '-private.json')
            write_state(private, {'records': rows})
            cells[cell] = rows
            receipts.append({'case_id': key, 'checkpoint_sha256': gate['checkpoint_sha256'],
                'private_paired_scores_sha256': file_sha256(private), 'full_state_readback_exact': True,
                'head_removed': True, 'model_unchanged': True, 'training_real_targets': case['end_targets'],
                'training_parameters': case['training_parameters'], 'deployed_parameters': config.parameter_count(True),
                'wall_seconds': time.time() - begin})
            del model
            gc.collect()
        per_seed[seed] = context_cells(cells, draws=plan['bootstrap_draws'], seed=plan['bootstrap_seed'])
        if time.time() - started > plan['cpu_seconds_cap']:
            raise TimeoutError('Fixed complete context-scoring/analysis CPU cap')
    guard()
    result = {'status': 'COMPLETE_EXPLORATORY_FOUR_CELL_NATURAL_CODE_CONTEXT', 'per_training_seed': per_seed,
        'model_receipts': receipts, 'forward_calls': calls, 'estimated_forward_matrix_ops': matrix,
        'parent_report_sha256': file_sha256(main_root / 'report.json'),
        'unchanged_primary_nll_decision': parent['decision'], 'panel_sha256': plan['panel_sha256'],
        'wall_seconds': time.time() - started, 'attempt_wall_seconds_including_wait': time.time() - ATTEMPT_STARTED,
        'assigned_chips': 0, 'optimizer_updates': 0,
        'test_accessed': False, 'terminal_alpha_spent': 0, 'terminal_success': False, 'scope': plan['scope']}
    write_state(out / 'report.json', result)
    write_state(PUBLIC / 'report.json', result)
    write_state(PUBLIC / 'status.json', {'status': result['status'], 'assigned_chips': 0})


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('CPU context observer interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as error:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--root', type=Path)
        parsed, _ = parser.parse_known_args()
        if parsed.root and (parsed.root / IDENTITY / 'plan.json').exists() and not (parsed.root / IDENTITY / 'report.json').exists():
            stopped = {'status': 'STOPPED_EXPLORATORY_FOUR_CELL_NATURAL_CODE_CONTEXT', 'error': str(error),
                'assigned_chips': 0, 'optimizer_updates': 0, 'terminal_success': False,
                'test_accessed': False, 'terminal_alpha_spent': 0,
                'forward_calls_through_failure': RUNTIME['forward_calls'],
                'estimated_forward_matrix_ops_through_failure': RUNTIME['estimated_forward_matrix_ops'],
                'cpu_phase_wall_seconds': time.time() - RUNTIME['compute_started_unix'] if 'compute_started_unix' in RUNTIME else 0,
                'attempt_wall_seconds_including_wait': time.time() - ATTEMPT_STARTED}
            write_state(parsed.root / IDENTITY / 'report.json', stopped)
            write_state(PUBLIC / 'report.json', stopped)
            write_state(PUBLIC / 'status.json', {'status': stopped['status'], 'assigned_chips': 0})
        raise
