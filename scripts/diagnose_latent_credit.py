"""No-update comparison of full and shortened Jacobi credit assignment."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time

import torch
from torch.nn import functional as F

from compute_trainium.indexed_natural_sources import IndexedNaturalSources
from compute_trainium.interleaved_model_factory import make_model
from compute_trainium.latent_credit_diagnostic import jacobi_credit, matrix_work, compare_gradients
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID = 'latent-exposure-credit-v1'
PUBLIC = Path('results/latent_exposure_20261030/credit-diagnostic-v1')
TRAINING = 'latent-learning-exposure-v1-latent'
MODES = ('all', 1, 0)


def run(root):
    started = time.time()
    out = root/ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every diagnostic attempt')
    training_path = root/TRAINING/'plan.json'
    training = json.loads(training_path.read_text())
    audit_path = root/TRAINING/'audit-latent/report.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
        raise AssertionError('Complete existing mature audit required')
    arm = training['arms']['latent']
    checkpoint = root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
    expected = audit['rows'][-1]['checkpoint_sha256']
    if training['seed'] != 730271 or arm['checkpoint_updates'] != [15287] or file_sha256(checkpoint) != expected:
        raise AssertionError('Only the fixed audited billion-target checkpoint')
    original_path = root/'interleaved-convergence-v1/plan.json'
    original = json.loads(original_path.read_text())
    files = list(Path('compute_trainium').glob('*.py')) + [Path(__file__).relative_to(Path.cwd()),
             Path('tests/test_latent_credit_diagnostic.py'), Path('docs/LATENT_CREDIT_DIAGNOSTIC_20261030.md')]
    plan = dict(identity=ID, registered_unix=started, absolute_deadline_unix=started+1200,
        checkpoint_sha256=expected, training_plan_sha256=file_sha256(training_path),
        audit_sha256=file_sha256(audit_path), original_window_plan_sha256=file_sha256(original_path),
        windows=original['windows'], rounds=[2, 3], retained_refinements=list(MODES),
        domain_weights={'web':.6, 'math':.2, 'code':.2}, threads=16,
        maximum_rss_bytes=16*1024**3, maximum_logit_error=.0002,
        source_sha256={str(f):file_sha256(f) for f in files},
        assigned_chips=0, optimizer_updates=0, heldout_reads=0, automatic_fit_admission=False,
        scope='TRAIN-only instantaneous gradient diagnostic. Forward identity does not '
              'establish equal optimization, quality or actual resource savings.')
    out.mkdir(mode=0o700); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'plan.json', plan)
    result = dict(status='STOPPED_LATENT_CREDIT_DIAGNOSTIC', rounds={})
    torch.set_num_threads(plan['threads'])

    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
        if (time.time() > plan['absolute_deadline_unix'] or available < 512*1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan['maximum_rss_bytes']
            or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3):
            raise RuntimeError('Credit diagnostic duration/resource guard')

    try:
        guard()
        data = IndexedNaturalSources(root/training['data_relative'])
        torch.manual_seed(training['seed'])
        model = make_model(training).eval()
        state = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); del state
        names = dict(model.named_parameters())
        per_domain = {d:sum(v['domain'] == d for v in plan['windows']) for d in plan['domain_weights']}
        for rounds in plan['rounds']:
            sums = {str(m):{n:torch.zeros_like(p, dtype=torch.float64) for n,p in names.items()} for m in MODES}
            rows = []
            for view in plan['windows']:
                guard()
                raw, _ = data.window(view['index'])
                if hashlib.sha256(raw.astype('<i8').tobytes()).hexdigest() != view['tokens_sha256']:
                    raise AssertionError('Frozen TRAIN window changed')
                x, y, mask = data.batch([view['index']])
                if not bool(mask.all()):
                    raise AssertionError('Existing diagnostic expects all-real targets')
                weight = plan['domain_weights'][view['domain']]/per_domain[view['domain']]
                reference, losses, errors = None, {}, {}
                for mode in MODES:
                    guard(); model.zero_grad(set_to_none=True)
                    logits = jacobi_credit(model, x, rounds, mode)
                    if reference is None:
                        reference = logits.detach().clone()
                    error = float((logits.detach()-reference).abs().max())
                    if error > plan['maximum_logit_error']:
                        raise AssertionError('Forward identity gate')
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                    loss.backward()
                    losses[str(mode)] = float(loss.detach())
                    errors[str(mode)] = error
                    for n,p in names.items():
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise AssertionError('Every parameter requires a finite gradient')
                        sums[str(mode)][n].add_(p.grad.double(), alpha=weight)
                    del logits, loss
                rows.append(dict(domain=view['domain'], index=view['index'],
                                 real_targets=int(mask.sum()), losses=losses, logit_errors=errors))
                del reference
                write_state(out/'progress.json', dict(rounds=rounds, completed_views=len(rows),
                                                     wall_seconds=time.time()-started))
            comparisons = {str(mode):compare_gradients(sums['all'], sums[str(mode)]) for mode in (1, 0)}
            result['rounds'][str(rounds)] = dict(rows=rows, gradient_comparisons=comparisons,
                matrix_work={str(m):matrix_work(model.config, 512, rounds, m) for m in MODES})
            del sums
            write_state(out/'progress-results.json', result)
        guard()
        if file_sha256(checkpoint) != expected or any(file_sha256(f) != h for f,h in plan['source_sha256'].items()):
            raise AssertionError('Frozen sources or checkpoint changed')
        result['status'] = 'COMPLETE_LATENT_CREDIT_DIAGNOSTIC'
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.time()-started, worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
            peak_rss_bytes=usage.ru_maxrss*1024, assigned_chips=0, optimizer_updates=0,
            heldout_reads=0, plan_sha256=file_sha256(out/'plan.json'), scope=plan['scope'])
        write_state(out/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'rounds'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
