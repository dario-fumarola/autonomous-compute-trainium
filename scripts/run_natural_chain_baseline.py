"""Before-curriculum readout of existing qualified models on a fixed new DEV."""
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import signal
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-chain-baseline-readout-v1'
PUBLIC=Path('results/natural_chain_baseline_20261025')


def register(root):
    import numpy as np
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import initial_state
    if managed_running():raise RuntimeError('Native ownership must be idle')
    prior=root/'natural-chain-driver-qualification-v3';qualified=json.loads((prior/'report.json').read_text());old=json.loads((prior/'plan.json').read_text())
    panel_root=root/'natural-chain-discovery-dev-v2';panel_report=json.loads((panel_root/'report.json').read_text())
    if qualified['status']!='PASS_NATURAL_CHAIN_DRIVER_QUALIFICATION' or panel_report['status']!='FROZEN_NATURAL_CHAIN_DISCOVERY_DEV':raise ValueError('Qualified task driver and adequate fixed DEV required')
    panel_file=panel_root/'panel.json'
    if file_sha256(panel_file)!=panel_report['panel_sha256']:raise AssertionError('New DEV changed')
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False);cfg=ParcaeCandidateConfig(**old['model']);panel=json.loads(panel_file.read_text());noise=[]
    for row in panel:
        prefix=row['views'][0]['prefix_ids'];seed=int(hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()[:16],16)%(2**63)
        noise.append(initial_state(cfg,0,seed)[:,:128].numpy()[0])
    np.save(out/'prompt-noise.npy',np.stack(noise));cases={}
    for seed in [5231,7283]:
        for arm in ['one','recurrent']:
            key=arm+'-s'+str(seed);folder=root/'runs'/f'future-span-main-v2-{arm}-ntp-s{seed}-lr32000-finish';report=json.loads((folder/'report.json').read_text());checkpoint=folder/'checkpoint-u1920.pt';expected=report['checkpoint_records'][-1]['checkpoint_sha256']
            if file_sha256(checkpoint)!=expected:raise AssertionError('Original NTP endpoint changed')
            cases[key]=dict(seed=seed,arm=arm,checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=expected,depths=[1] if arm=='one' else [1,2,4])
    extras=['scripts/run_natural_chain_baseline.py','scripts/read_natural_chain_native.py','scripts/audit_natural_chain_readout_cpu.py','compute_trainium/conclusion_generation.py','tests/test_conclusion_generation.py','docs/NATURAL_CHAIN_BASELINE_READOUT_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in extras}};worker,host=split_snapshot_sources(sources)
    plan=dict(identity=IDENTITY,registered_unix=time.time(),purpose='readonly_original_chain_baseline',source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        model=old['model'],cases=cases,panel=str(panel_file.relative_to(root)),panel_sha256=file_sha256(panel_file),noise=str((out/'prompt-noise.npy').relative_to(root)),noise_sha256=file_sha256(out/'prompt-noise.npy'),
        qualification_report_sha256=file_sha256(prior/'report.json'),seconds_limit=1800,phase_seconds_limit=600,cpu_phase_seconds_limit=600,
        root_comm_ports=reserve_ports(root,[IDENTITY+'-'+key for key in cases]),scope=Path(extras[-1]).read_text(),optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);return plan


def run(root,plan):
    out=root/IDENTITY;execution=Execution(root,out/'plan.json',out/'execution');records=[];result=dict(status='STOPPED_ORIGINAL_CHAIN_BASELINE');cpu=cf.ThreadPoolExecutor(2)
    def interrupted(signum,frame):execution.failed.set();raise KeyboardInterrupt('Readout interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def case(key):
        name=IDENTITY+'-'+key;opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',key,'--comm-port',str(plan['root_comm_ports'][name])]
        native=execution.native(name,'scripts.read_natural_chain_native',opts,'COMPLETE_READONLY_ORIGINAL_CHAIN_BASELINE')
        target=out/'cpu'/key;options=['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',execution.sha,'--case',key,'--capture',str(root/'runs'/name),'--output',str(target)]
        audit=cpu.submit(execution.cpu,key,'scripts.audit_natural_chain_readout_cpu',options,target/'report.json','PASS_INDEPENDENT_ORIGINAL_CHAIN_READOUT')
        return key,native,audit
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_READONLY_BEFORE_CURRICULUM_BASELINE',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as native_pool:
            for future in cf.as_completed([native_pool.submit(case,key) for key in plan['cases']]):
                key,native,audit=future.result();records.append(dict(case=key,native=native,independent_cpu=audit.result()))
                write_state(PUBLIC/'progress.json',dict(records=records))
        execution.check();result=dict(status='COMPLETE_ORIGINAL_CHAIN_BASELINE_READOUT',records=records)
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for sig,handler in handlers.items():signal.signal(sig,handler)
        result.update(plan_sha256=execution.sha,optimizer_updates=0,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0,scope=plan['scope']);write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root,register(a.root))


if __name__=='__main__':main()
