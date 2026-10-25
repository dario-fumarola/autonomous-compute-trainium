"""Finite scientific-driver recovery gate after the four-cell hardware pass."""
import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_scientific_state import INTEGRATION,matrix_work
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def wait_for_hardware(root):
    out=root/'future-span-driver-queue-v1';public=Path('results/future_span_driver_queue_20261025')
    qplan_file=root/'future-span-qualification-v1/plan.json';qplan=json.loads(qplan_file.read_text())
    files=['scripts/run_future_span_driver.py','scripts/train_future_span_scientific.py','scripts/audit_future_span_driver.py',
        'compute_trainium/future_span_execution.py','compute_trainium/future_span_scientific_state.py',
        'compute_trainium/parcae_scientific_state.py','compute_trainium/parcae_scientific_schedule.py',
        'tests/test_future_span_execution.py','tests/test_future_span_scientific_state.py','tests/test_future_span_driver_gate.py']
    plan_file=out/'plan.json'
    if plan_file.exists():saved=json.loads(plan_file.read_text())
    else:
        out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
        saved={'identity':'future-span-driver-queue-v1','registered_unix':time.time(),
            'source_sha256':{**qplan['source_sha256'],**{p:file_sha256(p) for p in files}},
            'hardware_plan_sha256':file_sha256(qplan_file),
            'latest_predecessor_finish_unix':qplan['predecessor_deadline_unix']+qplan['execution_seconds_limit'],
            'assigned_chips':0,'research_optimizer_updates':0,'test_accessed':False,
            'scope':'Wait for the complete independent hardware pass,then run only the separately bounded scientific-driver gate. Never claim a gap between phases or launch scientific tuning/main training.'}
        write_state(plan_file,saved);write_state(public/'plan.json',saved)
    if file_sha256(qplan_file)!=saved['hardware_plan_sha256']:raise AssertionError('Hardware prerequisite plan changed')
    report_file=root/'future-span-qualification-v1/report.json'
    while True:
        if any(file_sha256(p)!=sha for p,sha in saved['source_sha256'].items()):raise AssertionError('Queued driver source changed')
        if report_file.exists():
            report=json.loads(report_file.read_text())
            if report['status']!='PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION':
                raise RuntimeError('Hardware prerequisite stopped or failed;do not launch driver')
            if not managed_running():
                write_state(public/'status.json',{'status':'HARDWARE_COMPLETE_ENTERING_FINITE_DRIVER_GATE','assigned_chips':0})
                return
        if time.time()>saved['latest_predecessor_finish_unix']:raise TimeoutError('Original predecessor admission/execution window elapsed')
        write_state(public/'status.json',{'status':'WAITING_FOR_COMPLETE_INDEPENDENT_HARDWARE_QUALIFICATION','assigned_chips':0})
        time.sleep(10)


def register(root):
    hardware_file=root/'future-span-qualification-v1/report.json'
    hardware=json.loads(hardware_file.read_text())
    if hardware['status']!='PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION':raise ValueError('Complete independent hardware qualification required')
    qplan_file=root/'future-span-qualification-v1/plan.json';qplan=json.loads(qplan_file.read_text())
    if hardware['plan_sha256']!=file_sha256(qplan_file):raise AssertionError('Hardware report/plan identity differs')
    prep=root/qplan['preparation'];manifest=json.loads((prep/'manifest.json').read_text())
    if file_sha256(prep/'manifest.json')!=qplan['preparation_manifest_sha256']:raise AssertionError('Original qualification preparation changed')
    out=root/'future-span-driver-v1';public=Path('results/future_span_driver_20261025')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    files=['scripts/run_future_span_driver.py','scripts/train_future_span_scientific.py','scripts/audit_future_span_driver.py',
        'compute_trainium/future_span_execution.py','compute_trainium/future_span_scientific_state.py',
        'compute_trainium/parcae_scientific_state.py','compute_trainium/parcae_scientific_schedule.py',
        'tests/test_future_span_execution.py','tests/test_future_span_scientific_state.py']
    sources={**qplan['source_sha256'],**{p:file_sha256(p) for p in files}}
    if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Qualified/scientific source changed')
    config=ParcaeCandidateConfig(**manifest['model']);span=FutureSpanConfig(**manifest['future_config'])
    torch.set_num_threads(4)
    model=FutureSpanCandidate(config,span,initialize=False,fp16_body=True)
    optimizer_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon');del model
    packing=root/manifest['data']['packed'];meta=json.loads((packing/'manifest.json').read_text())
    if file_sha256(packing/'manifest.json')!=manifest['data']['manifest_sha256']:raise AssertionError('Natural packing changed')
    counts_file=packing/'prefix_counts.npy'
    if file_sha256(counts_file)!=meta['files']['prefix_counts.npy']:raise AssertionError('Original source counts changed')
    counts=np.load(counts_file,mmap_mode='r');anchor_prefix=np.load(prep/manifest['prefix_file'],mmap_mode='r')
    if file_sha256(prep/manifest['prefix_file'])!=manifest['prefix_sha256']:raise AssertionError('Qualified auxiliary counts changed')
    tables={}
    for arm in ['one','recurrent']:
        q=manifest['depths'][arm];path=prep/q['file']
        if file_sha256(path)!=q['sha256']:raise AssertionError('Qualified depth table changed')
        depth=np.array(np.load(path,mmap_mode='r')[:41],copy=True)
        work=matrix_work(config,span,depth,world=32,optimizer_per_replica=optimizer_ops)
        dp=out/(arm+'-depths.npy');wp=out/(arm+'-work.npy');np.save(dp,depth);np.save(wp,work)
        tables[arm]={'depth_table':dp.name,'depth_table_sha256':file_sha256(dp),
            'work_table':wp.name,'work_table_sha256':file_sha256(wp)}
    cases={};identities=[]
    for cell in qplan['cases'].values():
        key=cell['identity'];parent=key+'-parent';table=tables[cell['arm']]
        common={'cell':key,'arm':cell['arm'],'objective_weight':cell['objective_weight'],
            'seed':manifest['backbone_model_seed'],'state_seed':manifest['state_seed'],
            'initial_state':'qualification-original','training_parameters':manifest['training_parameters'],
            'peak_lr':1e-4,'warmup_targets':int(counts[10*512].sum()),'cooldown_start_update':20,
            'cooldown_start_targets':int(counts[20*512].sum()),**table}
        cases[parent]={**common,'total_updates':40,'end_targets':int(counts[40*512].sum()),
            'checkpoint_updates':[1,20,21,40],'segment_ends':[20,21,40]}
        cases[key+'-branch']={**common,'total_updates':30,'end_targets':int(counts[30*512].sum()),
            'checkpoint_updates':[30],'segment_ends':[30],'parent':{'case_id':parent,'update':20}}
        identities.extend(['future-span-driver-v1-'+key+'-'+p for p in ['first','finish','replay','branch']])
    initial={'file':'../'+prep.name+'/'+manifest['initial_file'],'sha256':manifest['initial_sha256'],
        'model_digest':manifest['model_digest'],'head_seed':manifest['head_seed']}
    plan={'identity':'future-span-driver-v1','registered_unix':time.time(),'purpose':INTEGRATION,
        'launch_gates_complete':True,'hardware_qualification_sha256':file_sha256(hardware_file),
        'source_sha256':sources,'worker_source_sha256':{p:sha for p,sha in sources.items() if p.startswith(('compute_trainium/','scripts/','configs/','third_party/'))},
        'model':manifest['model'],'future_config':manifest['future_config'],'gradient_rule':'all_sampled_steps',
        'precision':'fp16_body_fp32_heads_masters','accumulation':16,'routing_seed':manifest['routing_seed'],
        'data':manifest['data'],'auxiliary_data':{'directory':prep.name,'prefix_file':manifest['prefix_file'],'prefix_sha256':manifest['prefix_sha256']},
        'initial_states':{'qualification-original':initial},'cases':cases,
        'root_comm_ports':reserve_ports(root,identities),'seconds_limit':3600,'phase_seconds_limit':900,'cpu_phase_seconds_limit':600,
        'scope':'Scientific-driver qualification only after complete hardware pass. Each four-cell initializer runs20,continues40,replays21 and branches30 with its own cooldown:51 global updates per cell. Initial gradient matches retained independent512-window gold only after exact state/source/depth/noise matching. No held-out score or automatic scientific launch.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return out,public,plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--wait-hardware',action='store_true')
    args=parser.parse_args();out=args.root/'future-span-driver-v1';public=Path('results/future_span_driver_20261025')
    if args.wait_hardware:
        try:wait_for_hardware(args.root)
        except BaseException as error:
            receipt={'status':'STOPPED_BEFORE_FUTURE_DRIVER_EXECUTION','error':str(error),
                'assigned_chips':0,'research_optimizer_updates':0,'test_accessed':False}
            folder=args.root/'future-span-driver-queue-v1'
            if folder.exists():
                write_state(folder/'report.json',receipt)
                write_state(Path('results/future_span_driver_queue_20261025/report.json'),receipt)
                write_state(Path('results/future_span_driver_queue_20261025/status.json'),receipt)
            raise
    plan_file=out/'plan.json'
    if plan_file.exists():plan=json.loads(plan_file.read_text())
    else:out,public,plan=register(args.root)
    if args.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve a terminal driver attempt;explicit recovery identity required')
    execution=Execution(args.root,plan_file,out/'execution',resume=args.resume);plan_sha=execution.sha
    cpu_pool=cf.ThreadPoolExecutor(2);futures={};records=[]
    result={'status':'STOPPED_FUTURE_SPAN_SCIENTIFIC_DRIVER_INTEGRATION'}
    def run_cell(cell):
        key=cell['identity'];parent_case=key+'-parent';capture={}
        def fit(phase,case_id,end,parent=None):
            identity=plan['identity']+'-'+key+'-'+phase
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/future-span-driver-v1/plan.json',
                '--plan-sha',plan_sha,'--case',case_id,'--segment-end',str(end),
                '--comm-port',str(plan['root_comm_ports'][identity])]
            if parent:
                row=next(r for r in parent['checkpoint_records'] if r['update']==20)
                options+=['--resume','/work/instance-nvme/runs/'+capture['first']['identity']+'/checkpoint-u20.pt',
                          '--resume-sha',row['checkpoint_sha256']]
            report=execution.native(identity,'scripts.train_future_span_scientific',options,
                'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
            if (report['case_id']!=case_id or report['plan_sha256']!=plan_sha or report['update']!=end
                    or not report['qualification_only']):raise AssertionError('Driver phase identity differs')
            capture[phase]={'identity':identity,'report_sha256':file_sha256(args.root/'runs'/identity/'report.json')}
            return report
        first=fit('first',parent_case,20)
        fit('finish',parent_case,40,first);fit('replay',parent_case,21,first);fit('branch',key+'-branch',30,first)
        file=out/(key+'-captures.json');write_state(file,capture)
        target=out/'cpu'/key
        options=['--root',str(args.root),'--plan',str(plan_file),'--plan-sha',plan_sha,'--cell',key,
            '--captures',str(file),'--output',str(target),'--upstream',str(args.root/'upstream-parcae-69284c1')]
        return cpu_pool.submit(execution.cpu,key+'-audit','scripts.audit_future_span_driver',options,target/'report.json',
            'PASS_FUTURE_SPAN_SCIENTIFIC_DRIVER_INTEGRATION')
    def guarded_cell(cell):
        try:return run_cell(cell)
        except BaseException:execution.failed.set();raise
    try:
        cells=[{'identity':key,**plan['cases'][key+'-parent']} for key in ['one-ntp','one-future','recurrent-ntp','recurrent-future']]
        with cf.ThreadPoolExecutor(2) as pool:
            for cell,future in zip(cells,[pool.submit(guarded_cell,cell) for cell in cells],strict=True):
                futures[cell['identity']]=future.result()
        for key,future in futures.items():
            records.append(future.result(timeout=max(1,execution.deadline-time.time())))
            write_state(public/'progress.json',{'records':records})
        execution.check();result={'status':'PASS_COMPLETE_FUTURE_SPAN_SCIENTIFIC_DRIVER','records':records,
            'scientific_training_launched':False,'scope':plan['scope']}
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records);raise
    finally:
        execution.failed.set();cpu_pool.shutdown(wait=True,cancel_futures=True)
        execution.close(public/'execution-costs.json')
        result.update(plan_sha256=plan_sha,qualification_only=True,research_optimizer_updates=0,test_accessed=False)
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})


if __name__=='__main__':main()
