"""Two separately admitted numerical stages for independent output/value tables."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.model import ModelConfig
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

BASE='probability-untied'
PUBLIC=Path('results/probability_workspace_20261029')


def paths(root,stage):
    identity=f'{BASE}-{stage}-v1'
    return identity,root/identity,PUBLIC/f'untied-{stage}-v1'


def prepare(root,stage):
    identity,out,public=paths(root,stage)
    if managed_running():raise RuntimeError('Previous Native owner remains active')
    if out.exists() or public.exists():raise FileExistsError('Preserve previous registration')
    small=stage=='numerical';allowance=(1 if small else 4)*1024**3
    if shutil.disk_usage('/').free<35*1024**3 or shutil.disk_usage(root).free<35*1024**3+allowance:
        raise RuntimeError('Protected disk floor plus complete artifact allowance')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete projected inode reserve')
    prerequisites={}
    if not small:
        gate=paths(root,'numerical')[2]/'report.json'
        if json.loads(gate.read_text())['status']!='PASS_UNTIED_PROBABILITY_NUMERICAL':raise AssertionError('Small fixture first')
        prerequisites[str(gate)]=file_sha256(gate)
        source=root/'probability-workspace-data-v1';r=json.loads((source/'report.json').read_text())
        if r['status']!='PASS_PROBABILITY_WORKSPACE_NATURAL_SOURCES' or r['reserved_group_overlap'] or r['source_group_overlap']:
            raise AssertionError('Natural source isolation')
        if any(file_sha256(source/n)!=h for n,h in r['files'].items()):raise AssertionError('Source bytes changed')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/UNTIED_PROBABILITY_INTERFACE_20261029.md','tests/test_untied_probability_workspace.py'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[identity+'-'+m for m in ('warm','replay')]
    config=(ModelConfig(dim=128,layers=2,heads=4,kv_heads=2,ffn_dim=256,vocab_size=1024,max_seq=64) if small else
            ModelConfig(dim=512,layers=16,heads=8,kv_heads=4,ffn_dim=1376,vocab_size=32000,max_seq=512))
    now=time.time()
    p=dict(identity=identity,purpose='probability_workspace_software_qualification' if small else 'probability_workspace_actual_shape_qualification',
        registered_unix=now,seconds_limit=3600,phase_seconds_limit=1500,cpu_phase_seconds_limit=1200,
        absolute_deadline_unix=now+5400,clock_admission_waiver=True,world=64,worker_memory_gib=128 if small else 192,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,prerequisites=prerequisites,
        root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=m) for n,m in zip(names,('warm','replay'))},
        inode_projection=projection,artifact_allowance_bytes=allowance,model_config=config.__dict__,
        parameters=config.parameter_count()+config.vocab_size*config.dim,untied_prediction_table=True,checkpoint_rank_batch=4,
        seed=9317 if small else 9419,qualification_only=True,retained_for_science=False,scientific_fit_admitted=False,
        protected_evaluation_groups_scored=0,research_optimizer_updates=0,
        scope='Disposable numerical qualification only. Existing 2% gradient/update/moment and .003 loss rules unchanged; separately gate input-table,output-table and body gradients. No scientific quality scoring or fitting follows automatically.')
    if small:p.update(length=32,update_cases=[[0,'expected'],[1,'expected'],[2,'expected'],[2,'uniform'],[1,'expected']],research_examples=0)
    else:p.update(data_relative='probability-workspace-data-v1',data_report_sha256=file_sha256(source/'report.json'),
        lr=3e-4,batch=4,early_warm_mean_seconds_limit=10,warm_phase=names[0],cold_local_supported_steps=[0,1,2],
        warm_updates=100,initial_training_updates=3,replay_update=104,
        training_schedule='extra_steps=(update-1)%3 through103;104 uses2 for exact replay/mature audit',
        artifact_basis='Two complete79.18M FP32 model/AdamW states plus two FP32 gradient captures total under2.5GiB;4GiB allowance includes source/receipts/compiler artifacts. Existing35GiB floor unchanged.')
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',p);write_state(public/'plan.json',p)


def run(root,stage):
    identity,out,public=paths(root,stage);small=stage=='numerical'
    e=ScreenExecution(root,out/'plan.json',out/'execution')
    r=dict(status='RUNNING_UNTIED_PROBABILITY_QUALIFICATION',records={},scientific_fit_admitted=False)
    try:
        for mode in ('warm','replay'):
            name=identity+'-'+mode
            args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{identity}/plan.json',
                  '--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name]),'--mode',mode]
            if mode=='replay':args+=['--parent',f'/work/instance-nvme/runs/{identity}-warm']
            module='scripts.qualify_probability_workspace' if small else 'scripts.qualify_probability_warm'
            status=('PASS_PROBABILITY_WORKSPACE_SOFTWARE_FIXTURE' if small else
                    'COMPLETE_PROBABILITY_WARM_PENDING_GLOBAL_AUDIT' if mode=='warm' else 'PASS_PROBABILITY_WARM_EXACT_REPLAY')
            r['records'][mode]=e.native(name,module,args,status);write_state(public/'progress.json',r)
        if not small:
            dest=out/'cpu-audit'
            r['records']['audit']=e.cpu(identity+'-cpu','scripts.audit_probability_warm',
                ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--output',str(dest)],
                dest/'report.json','PASS_PROBABILITY_MATURE_GLOBAL_CPU')
        r['status']='PASS_UNTIED_PROBABILITY_NUMERICAL' if small else 'PASS_UNTIED_PROBABILITY_ACTUAL_SHAPE'
    except BaseException as error:r.update(status='STOPPED_UNTIED_PROBABILITY_QUALIFICATION',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(public/'report.json',r);e.close(public/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['numerical','warm'],required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root,a.stage)
