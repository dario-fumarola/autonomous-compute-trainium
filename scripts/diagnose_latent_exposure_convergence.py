"""Repeat the existing TRAIN consistency diagnostic at the frozen 1B endpoint."""
import argparse
import copy
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import diagnose_interleaved_convergence as diagnostic

ID = 'latent-exposure-convergence-v1'
TUNING = 'latent-learning-exposure-v1-latent'
PUBLIC = Path('results/latent_exposure_20261030/convergence-v1')


def prepare(root):
    old_path = root/'interleaved-convergence-v1/plan.json'
    old = json.loads(old_path.read_text())
    training_path = root/TUNING/'plan.json'
    training = json.loads(training_path.read_text())
    audit_path = root/TUNING/'audit-latent/report.json'
    audit = json.loads(audit_path.read_text())
    if (set(training['arms']) != {'latent'} or training['seed'] != 730271
            or training['arms']['latent']['checkpoint_updates'] != [15287]
            or audit['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS'):
        raise AssertionError('Only the completed, audited billion-target endpoint')
    if root.joinpath(ID).exists() or PUBLIC.exists():
        raise FileExistsError('Preserve all diagnostic attempts')
    p = copy.deepcopy(old)
    files = list(Path('compute_trainium').glob('*.py')) + [
        Path(__file__).relative_to(Path.cwd()),
        Path('scripts/diagnose_interleaved_convergence.py'),
        Path('scripts/qualify_native.py'),
        Path('docs/LATENT_EXPOSURE_CONVERGENCE_20261030.md')]
    now = time.time()
    p.update(identity=ID, registered_unix=now, absolute_deadline_unix=now+1200,
        endpoint_seconds_limit=1200, arms=['latent'],
        tuning_plan_sha256=file_sha256(training_path),
        original_window_plan_sha256=file_sha256(old_path),
        completed_audit_sha256=file_sha256(audit_path),
        source_sha256={str(f): file_sha256(f) for f in files},
        scope='Existing six TRAIN windows and numerical tolerances at the audited '
              '1B endpoint. No quality selection, optimizer updates, new inference '
              'thought budgets, reserve scoring or automatic training admission.')
    (root/ID).mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json', p)
    write_state(PUBLIC/'plan.json', p)
    print(json.dumps(dict(status='REGISTERED_LATENT_EXPOSURE_CONVERGENCE',
                         windows=len(p['windows']), assigned_chips=0)))


def run(root):
    diagnostic.ID, diagnostic.TUNING, diagnostic.PUBLIC = ID, TUNING, PUBLIC
    diagnostic.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare', 'run'], required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.root)
