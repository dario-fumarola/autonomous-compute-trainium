"""Recover a pre-update inode exhaustion within the original pilot deadline."""
import argparse
import copy
import json
import os
from pathlib import Path
from compute_trainium.future_span_execution import Execution
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_natural_chain_pilot as original

IDENTITY='natural-chain-continuation-pilot-v2'
PUBLIC=Path('results/natural_chain_continuation_pilot_20261025/recovery-v2')

class RecoveredExecution(Execution):
    def __init__(self,root,plan_file,out):super().__init__(root,plan_file,out,resume=True)
    def check(self):
        super().check();require_inode_reserve([self.root,Path('/')])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();old=a.root/'natural-chain-continuation-pilot-v1';plan=json.loads((old/'plan.json').read_text());clock=json.loads((old/'execution/execution-start.json').read_text())
    log=a.root/'natural-chain-continuation-pilot-v1-controller.log'
    if 'No space left on device' not in log.read_text():raise ValueError('Unexplained predecessor failure')
    for case in plan['cases']:
        for mode in ['preflight','fit']:
            folder=a.root/'runs'/('natural-chain-continuation-pilot-v1-'+case+'-'+mode)
            if any(folder.glob('*.pt')) or (folder/'report.json').exists():raise ValueError('Unexpected fitted state;no blind restart')
    failure=dict(status='STOPPED_PREUPDATE_FILESYSTEM_INODE_EXHAUSTION',plan_sha256=file_sha256(old/'plan.json'),actual_optimizer_updates=0,research_optimizer_updates=0,native_entry_launch_receipts=0,scope='First supervised source snapshot failed before its launch receipt;second phase failed opening its log. No native model/gradient/checkpoint artifacts. Both owned workers canceled. Original controller log and partial snapshot preserved. Bytecode-only cleanup restores130930 inodes;no source/model/optimizer/NEFF/HLO removed.')
    write_state(old/'report.json',failure);write_state(Path('results/natural_chain_continuation_pilot_20261025/report.json'),failure)
    out=a.root/IDENTITY;out.mkdir()
    for name in ['initial-body.pt','tasks.pt']:(out/name).symlink_to(os.path.relpath(old/name,out))
    additions=['scripts/recover_natural_chain_pilot.py','compute_trainium/inode_reserve.py','tests/test_inode_reserve.py']
    sources={**plan['source_sha256'],**{p:file_sha256(p) for p in additions}};worker,host=split_snapshot_sources(sources)
    names=[name.replace('natural-chain-continuation-pilot-v1',IDENTITY,1) for name in plan['root_comm_ports']]
    new=copy.deepcopy(plan);new.update(identity=IDENTITY,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(a.root,names),original_deadline_unix=clock['deadline_unix'],failed_plan_sha256=file_sha256(old/'plan.json'),
        failed_report_sha256=file_sha256(old/'report.json'),recovery_scope='Only new owned worker identities and host inode reserve guard after bytecode-only cleanup. Identical body/task assets,source order,LR,depths,objectives,endpoints,work controls,numerical gates and original absolute deadline. No native update occurred in the failed admission;its reservation opportunity remains counted without double addition.')
    PUBLIC.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',new);write_state(PUBLIC/'plan.json',new);(out/'execution').mkdir();write_state(out/'execution/execution-start.json',{**clock,'plan_sha256':file_sha256(out/'plan.json')})
    original.IDENTITY=IDENTITY;original.PUBLIC=PUBLIC;original.Execution=RecoveredExecution;original.run(a.root,new)


if __name__=='__main__':main()
