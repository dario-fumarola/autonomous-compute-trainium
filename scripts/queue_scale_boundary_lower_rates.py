"""Wait for exact parent closure, then admit only the registered rate check."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_scale_boundary_lower_rates as stage


def main(root):
    folder = root / 'scale-boundary-lower-rate-queue-v1'
    if folder.exists():
        raise FileExistsError('Preserve every conditional queue')
    parent = root / stage.PARENT
    scope = root / stage.SCOPE / 'scope.json'
    plan = dict(registered_unix=time.time(), operational_wait_seconds=6 * 3600,
                scope_sha256=file_sha256(scope), parent_plan_sha256=file_sha256(parent / 'plan.json'),
                parent_pid=int((parent / 'controller.pid').read_text()),
                source_sha256=file_sha256(Path(__file__)), scientific_main_fit_admitted=False)
    folder.mkdir(); write_state(folder / 'plan.json', plan)
    result = dict(status='STOPPED_CONDITIONAL_LOWER_RATE_QUEUE', scientific_main_fit_admitted=False)
    try:
        while True:
            if (file_sha256(scope) != plan['scope_sha256']
                    or file_sha256(parent / 'plan.json') != plan['parent_plan_sha256']):
                raise AssertionError('Frozen parent or prospective scope changed')
            if time.time() > plan['registered_unix'] + plan['operational_wait_seconds']:
                raise TimeoutError('Bounded operational wait ended; no automatic renewal')
            if (parent / 'report.json').exists():
                report = json.loads((parent / 'report.json').read_text())
                if report['status'] != 'COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION':
                    raise ValueError('Original stage stopped; no automatic continuation')
                # Its bounded archival operation must close before a new
                # Native artifact allocation replaces the declared parent.
                archive = root / 'closed-legacy-checkpoint-exact-archive-v1'
                archive_closed = not archive.exists() or (archive / 'report.json').exists()
                if not managed_running() and archive_closed:
                    break
            elif subprocess.run(['kill', '-0', str(plan['parent_pid'])], capture_output=True).returncode:
                raise RuntimeError('Parent vanished without a terminal receipt')
            time.sleep(10)
        admitted = stage.prepare(root)
        if admitted:
            stage.run(root)
        result.update(status='COMPLETE_CONDITIONAL_LOWER_RATE_QUEUE', additional_rates_needed=admitted,
                      followup_report_sha256=file_sha256(root / stage.IDENTITY / 'report.json'))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        result['wall_seconds'] = time.time() - plan['registered_unix']
        write_state(folder / 'report.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
