"""Permission-only recovery after a zero-mutation consolidation stop."""
import argparse
import json
from pathlib import Path
import time

from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts import deduplicate_closed_recovery_copies as original


def main(root):
    parent=root/original.ID
    report=json.loads((parent/'report.json').read_text())
    if (report['status']!='STOPPED_CLOSED_RECOVERY_COPY_CONSOLIDATION'
            or report['error_type']!='PermissionError' or report['files']!=0 or report['freed_bytes']!=0):
        raise AssertionError('Only the recorded zero-mutation permission stop')
    public=Path('results/closed_recovery_copy_consolidation_20261030/permission-recovery-v1')
    admission=public.parent/'permission-recovery-admission.json'
    if admission.exists(): raise FileExistsError('Preserve recovery attempts')
    write_state(admission,dict(registered_unix=time.time(),parent_report_sha256=file_sha256(parent/'report.json'),
        parent_plan_sha256=file_sha256(parent/'PRIVATE-plan.json'),
        source_sha256={f:file_sha256(f) for f in ('scripts/recover_closed_copy_consolidation.py',
                                               'scripts/deduplicate_closed_recovery_copies.py')},
        scope='Run the same six-copy full-hash verification and path-preserving consolidation with directory-owner permissions. '
              'No prior files changed. The parent aggregate\'s redundancy-reduced boolean described intended scope; '
              'its files=0 and freed_bytes=0 establish that no redundancy was actually reduced in that failed attempt.'))
    original.ID='closed-recovery-copy-consolidation-permission-v1'
    original.PUBLIC=public
    original.main(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
