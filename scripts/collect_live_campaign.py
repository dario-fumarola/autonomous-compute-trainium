"""Read-only observations of immutable launches, updates and closed receipts."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state


def collect(root, identity, public):
    now = time.time()
    opened, closed = [], []
    for snapshot in sorted((root/'snapshots').glob(identity+'-*')):
        launch = json.loads((snapshot/'launch.json').read_text())
        closure = snapshot/'closure.json'
        if closure.exists():
            closed.append({'identity': snapshot.name, **json.loads(closure.read_text())})
            continue
        log = root/'logs'/(snapshot.name+'.log')
        updates = []
        for line in log.read_text(errors='replace').splitlines() if log.exists() else []:
            try: row = json.loads(line)
            except ValueError: continue
            if 'update' in row and 'gradient_norm' in row:
                updates.append(row)
        item = {'identity': snapshot.name, 'assigned_chips': len(launch['physical_chips']),
                'observed_launch_seconds': now-launch['started_unix'],
                'last_logged_training_update': updates[-1] if updates else None}
        if len(updates) >= 22:
            first, last = updates[-21], updates[-1]
            item['rolling200_update_targets_per_second'] = (
                (last['real_tokens']-first['real_tokens'])/(last['elapsed_seconds']-first['elapsed_seconds']))
        opened.append(item)
    terminal = root/identity/'report.json'
    result = {'status': 'CLOSED_OBSERVATION' if terminal.exists() else 'LIVE_OBSERVATION',
              'observed_unix': now, 'closed_attempts': closed, 'open_attempts': opened,
              'closed_assigned_chip_hours': sum(r['assigned_chip_hours'] for r in closed),
              'open_assigned_chip_hours_so_far': sum(r['assigned_chips']*r['observed_launch_seconds']/3600 for r in opened),
              'scope': 'Read-only receipt/update observations, not new training or held-out evidence. '
                       'Rank-zero logged NLL is not aggregate DEV. Rolling rates are not controlled '
                       'performance qualification. Other campaigns overlap; do not add their wall '
                       'intervals as sequential reservation time. CPU work has separate receipts.'}
    if terminal.exists(): result['campaign_status'] = json.loads(terminal.read_text())['status']
    write_state(public/'live-observation.json', result)
    return terminal.exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--follow', action='store_true')
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    if Path(a.identity).name != a.identity:
        raise ValueError('Single campaign identity required')
    a.output.mkdir(exist_ok=True)
    while True:
        closed = collect(a.root, a.identity, a.output)
        if closed or not a.follow or time.time() > a.deadline_unix:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
