"""Retire eight named successful, disposable qualification arrays only."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(root):
    start = time.time()
    if subprocess.check_output(['docker', 'ps', '-q', '--filter',
                                'label=ac-trainium-managed=1']).strip():
        raise RuntimeError('All accelerator owners must be closed')
    out = root / 'closed-probability-qualification-retirement-v2'
    if out.exists():
        raise FileExistsError('No automatic maintenance repetition')
    gates, files = {}, []
    for name, status in [('joint-exit', 'PASS_JOINT_EXIT_PROBABILITY_ACTUAL_SHAPE'),
                         ('untied', 'PASS_UNTIED_PROBABILITY_ACTUAL_SHAPE')]:
        gate = Path(f'results/probability_workspace_20261029/{name}-warm-v1/report.json')
        if json.loads(gate.read_text())['status'] != status:
            raise AssertionError('Only completed successful qualifications may be retired')
        gates[str(gate)] = digest(gate)
        for phase, names in [('warm', ['PRIVATE-u103.pt', 'PRIVATE-u104.pt',
                                      'PRIVATE-mature-gradient.pt']),
                             ('replay', ['PRIVATE-mature-gradient.pt'])]:
            folder = root / 'runs' / f'probability-{name}-warm-v1-{phase}'
            for filename in names:
                path = folder / filename
                if not path.is_file() or path.is_symlink():
                    raise AssertionError(f'Original named array required: {path.name}')
                files.append(dict(path=str(path.relative_to(root)),
                                  bytes=path.stat().st_size, sha256=digest(path)))
    scope = ('Only eight disposable successful qualification arrays for the closed '
             'untied and joint-exit probability recipes. All scientific checkpoints, '
             'failed histories, original receipts, sources and canonical small '
             'numerical fixtures remain. Exact local restoration of these eight '
             'arrays becomes unavailable; source/seed/configuration permit '
             'recomputation, without a bit-exact promise.')
    plan = dict(registered_unix=start, scope=scope, files=files, gate_sha256=gates,
                seconds_limit=600, exact_local_restoration_after_retirement=False)
    out.mkdir()
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    before = shutil.disk_usage(root).free
    for row in files:
        path = root / row['path']
        if time.time() - start > 600 or digest(path) != row['sha256']:
            raise RuntimeError('Maintenance limit or changed source array')
        receipt = dict(**row, scope=scope, plan=out.name + '/plan.json',
                       exact_local_restoration_available=False)
        path.with_name(path.name + '.retired.json').write_text(json.dumps(receipt, indent=2) + '\n')
        path.unlink()
    report = dict(status='COMPLETE_CLOSED_QUALIFICATION_RETIREMENT', **plan,
                  freed_bytes=sum(row['bytes'] for row in files), free_before=before,
                  free_after=shutil.disk_usage(root).free, wall_seconds=time.time()-start,
                  scientific_checkpoints_deleted=0, failed_histories_deleted=0,
                  assigned_chips=0)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'freed_bytes', 'free_after', 'wall_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
