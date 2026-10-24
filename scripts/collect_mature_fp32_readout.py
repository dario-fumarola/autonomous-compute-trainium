"""Read-only publication of completed independent audits during native fits."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--follow',action='store_true')
    a=p.parse_args()
    campaign=a.root/'mature-fp32-v1'
    public=Path('results/mature_fp32_20261024')
    plan=json.loads((campaign/'plan.json').read_text())
    while True:
        progress=campaign/'independent-cpu/progress.json'
        if progress.exists():
            raw=progress.read_bytes();snapshot=json.loads(raw)
            rows=snapshot['records']
            if any(not row['copy_checksum_and_full_state_readback'] or abs(row['cpu_minus_native_nll'])>.0002 for row in rows):
                raise AssertionError('Only completed passing independent audit rows may be published')
            write_state(public/'live-independent-cpu.json',{'status':'ALL_SIX_AUDIT_ROWS_COMPLETED' if len(rows)==6 else 'PARTIAL_COMPLETED_AUDIT_ROWS',
                'observed_unix':time.time(),'records':rows,'expected_records':6,
                'private_progress_sha256':hashlib.sha256(raw).hexdigest(),
                'scope':'Read-only copy of completed whole-DEV independent FP32 scores and full-state serialized recovery checks. No new fit, source change or TEST. Official campaign closure remains separate.'})
        if not a.follow or (campaign/'report.json').exists():return
        if time.time()>plan['hard_deadline_unix']+300:raise TimeoutError('Readout collector deadline')
        time.sleep(20)


if __name__=='__main__':main()
