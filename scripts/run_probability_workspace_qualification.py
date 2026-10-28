"""Register and execute only the bounded small-array Native numerical fixture."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from compute_trainium.model import ModelConfig
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution


def main(root, action):
    identity='probability-workspace-numerical-v1';out=root/identity
    public=Path('results/probability_workspace_20261028/numerical-v1')
    if action=='prepare':
        from compute_trainium.execution_isolation import managed_running
        if managed_running():raise RuntimeError('Previous owner still active')
        for f in (root,Path('/')):
            if shutil.disk_usage(f).free<37*1024**3:raise RuntimeError('35GiB floor plus 2GiB qualification allowance')
        inode=project_snapshot_inodes(Path('.'),2)
        import os
        if os.statvfs(root).f_favail<inode['required_free_inodes']:raise RuntimeError('Complete snapshot inode allowance required')
        sources={str(f):file_sha256(f) for directory in ('compute_trainium','scripts','configs') for f in Path(directory).rglob('*.py')}
        for f in ('docs/PROBABILITY_WORKSPACE_20261028.md','tests/test_probability_workspace.py','scripts/run_native_isolated.sh'):
            sources[f]=file_sha256(f)
        worker,host=split_snapshot_sources(sources);names=[identity+'-'+m for m in ('warm','replay')]
        plan=dict(identity=identity,purpose='probability_workspace_software_qualification',registered_unix=time.time(),
            world=64,worker_memory_gib=128,seconds_limit=1800,phase_seconds_limit=1200,
            clock_admission_waiver=True,absolute_deadline_unix=time.time()+3600,
            source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
            root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=m) for n,m in zip(names,('warm','replay'))},
            model_config=asdict(ModelConfig(dim=128,layers=2,heads=4,kv_heads=2,ffn_dim=256,vocab_size=1024,max_seq=64)),
            length=32,seed=9317,update_cases=[[0,'expected'],[1,'expected'],[2,'expected'],[2,'uniform'],[1,'expected']],
            artifact_allowance_bytes=2*1024**3,inode_projection=inode,scientific_fit_admitted=False,
            research_examples=0,research_optimizer_updates=0)
        out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
        return
    execution=AllChipExecution(root,out/'plan.json',out/'execution')
    result=dict(status='STOPPED_PROBABILITY_WORKSPACE_NUMERICAL',records={},scientific_fit_admitted=False)
    try:
        for mode in ('warm','replay'):
            name=identity+'-'+mode
            options=['--plan','/work/instance-nvme/'+identity+'/plan.json','--plan-sha',execution.sha,
                     '--comm-port',str(execution.plan['root_comm_ports'][name]),'--mode',mode]
            if mode=='replay':options+=['--parent','/work/instance-nvme/runs/'+identity+'-warm']
            result['records'][mode]=execution.native(name,'scripts.qualify_probability_workspace',options,'PASS_PROBABILITY_WORKSPACE_SOFTWARE_FIXTURE')
            write_state(public/'progress.json',result)
        result['status']='PASS_PROBABILITY_WORKSPACE_NUMERICAL_AND_REPLAY'
    except BaseException as e:
        result.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        write_state(public/'report.json',result);execution.close(public/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();main(a.root,a.action)
