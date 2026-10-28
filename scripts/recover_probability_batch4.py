"""Complete only the unstarted replay/audit after a host-admission stop."""
import argparse,json,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution

ID='probability-workspace-batch4-recovery-v1'
PUBLIC=Path('results/probability_workspace_20261028/batch4-recovery-v1')

def main(root):
    old=root/'probability-workspace-batch4-v1';out=root/ID
    p=json.loads((old/'plan.json').read_text());parent=root/'runs'/p['warm_phase']
    if json.loads((parent/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_WARM_PENDING_GLOBAL_AUDIT':raise AssertionError('Completed warm phase required')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    files['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
    worker,host=split_snapshot_sources(files);name=ID+'-replay'
    p.update(identity=ID,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        registered_unix=time.time(),absolute_deadline_unix=time.time()+3600,seconds_limit=3600,
        root_comm_ports=reserve_ports(root,[name]),phases={name:dict(mode='replay')},
        original_plan_sha256=file_sha256(old/'plan.json'),recovery_reason='Host allowance admission stopped before replay launch; no numerical failure, no replacement warm updates.',
        parent_states={n:file_sha256(parent/n) for n in ('PRIVATE-u103.pt','PRIVATE-u104.pt','PRIVATE-mature-gradient.pt')})
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    e=AllChipExecution(root,out/'plan.json',out/'execution');r=dict(status='STOPPED_BATCH4_RECOVERY',records={})
    try:
        r['records']['replay']=e.native(name,'scripts.qualify_probability_warm',[
            '--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,
            '--mode','replay','--parent','/work/instance-nvme/runs/'+p['warm_phase'],
            '--comm-port',str(p['root_comm_ports'][name])],'PASS_PROBABILITY_WARM_EXACT_REPLAY')
        dest=out/'cpu-audit'
        r['records']['audit']=e.cpu(ID+'-cpu','scripts.audit_probability_warm',[
            '--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(dest)],
            dest/'report.json','PASS_PROBABILITY_MATURE_GLOBAL_CPU')
        if any(file_sha256(parent/n)!=h for n,h in p['parent_states'].items()):raise AssertionError('Warm states changed')
        r['status']='PASS_PROBABILITY_BATCH4_RECOVERY_AND_MATURE_AUDIT'
    except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root)
