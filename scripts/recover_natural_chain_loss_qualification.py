"""Same-deadline explicit-CE recovery of the native graph break."""
import argparse
import copy
import json
import os
from pathlib import Path
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_natural_chain_qualification as original

IDENTITY='natural-chain-driver-qualification-v3'
PUBLIC=Path('results/natural_chain_driver_qualification_20261025/recovery-v3')

class RecoveredExecution(Execution):
    def __init__(self,root,plan_file,out):super().__init__(root,plan_file,out,resume=True)
    def native(self,identity,module,options,status):
        if module=='scripts.qualify_natural_chain_native':module='scripts.qualify_natural_chain_native_v3'
        return super().native(identity,module,options,status)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();old=a.root/'natural-chain-driver-qualification-v2'
    plan=json.loads((old/'plan.json').read_text());failure=json.loads((old/'report.json').read_text());clock=json.loads((old/'execution/execution-start.json').read_text())
    if failure['status']!='STOPPED_NATURAL_CHAIN_DRIVER_QUALIFICATION' or failure['records']:raise ValueError('Only preserved pre-update CLI failure eligible')
    for case in ['one','recurrent']:
        run=a.root/'runs'/('natural-chain-driver-qualification-v2-'+case+'-warm')
        if any(run.glob('*.pt')) or (run/'report.json').exists():raise ValueError('Unexpected native model/update artifact;do not replay blindly')
        log=a.root/'logs'/('natural-chain-driver-qualification-v2-'+case+'-warm.log')
        if 'DataDependentOutputException: aten.cross_entropy_loss.default' not in log.read_text():raise ValueError('Unexplained failure;recovery not admitted')
    out=a.root/IDENTITY;out.mkdir()
    for name in ['initial-body.pt','tasks.pt']:(out/name).symlink_to(os.path.relpath(old/name,out))
    sources={**plan['source_sha256'],**{p:file_sha256(p) for p in ['compute_trainium/natural_chain_native_loss.py','scripts/qualify_natural_chain_native_v3.py','scripts/recover_natural_chain_loss_qualification.py','tests/test_natural_chain_native_loss.py']}}
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-'+k+'-'+m for k in ['one','recurrent'] for m in ['warm','replay']]
    new=copy.deepcopy(plan);new.update(identity=IDENTITY,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(a.root,names),failed_plan_sha256=file_sha256(old/'plan.json'),failed_report_sha256=file_sha256(old/'report.json'),
        original_deadline_unix=clock['deadline_unix'],recovery_scope='Replace unsupported native cross_entropy by the existing qualified explicit full-vocabulary logsumexp/gather formula,with CPU value/gradient/AOT parity. Keep the earlier CLI adapter. Same source views,weights,model,loss,optimizer,schedule,counts,numerical gates and original absolute deadline. Failed assigned costs remain counted;reservation intervals overlap.')
    PUBLIC.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',new);write_state(PUBLIC/'plan.json',new)
    (out/'execution').mkdir();write_state(out/'execution/execution-start.json',{**clock,'plan_sha256':file_sha256(out/'plan.json')})
    original.IDENTITY=IDENTITY;original.PUBLIC=PUBLIC;original.Execution=RecoveredExecution;original.run(a.root,new)


if __name__=='__main__':main()
