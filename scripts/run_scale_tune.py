"""Frozen independent rate sweeps; no automatic larger scientific comparison."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import AllChipExecution,SCALE_START,SCALE_END

IDENTITY='scale-medium-tune-v1'
PUBLIC=Path('results/scale_medium_tune_20261027')


def prepare(root):
    import numpy as np
    from compute_trainium.scale_pilot_size import profiled_inventory
    from compute_trainium.scale_work_ledger import fixed_update_work
    if managed_running():raise RuntimeError('Prior Native owners must be terminal')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve all rate-sweep registrations')
    gate=root/'scale-training-state-gate-v1/report.json'
    if json.loads(gate.read_text())['status']!='PASS_SCALE_REGISTERED_TRAINER_AND_REPLAY':raise ValueError('Complete training state gate required')
    source_audit=root/'scale-capacity-reverification-v1-recovery/report.json'
    if json.loads(source_audit.read_text())['status']!='REVERIFIED_AUTHENTIC_SCALE_CAPACITY_NO_MODEL_SOURCE_MIGRATION':raise ValueError('Current source proof required')
    numerical={}
    for native,status in [('scale-376m-fixed-two-numerical-v1','PASS_SCALE_MEDIUM_FIXED_TWO_COMPLETE_NUMERICAL_QUALIFICATION'),
                          ('scale-375m-ordinary-numerical-v1','PASS_SCALE_MEDIUM_ORDINARY_COMPLETE_NUMERICAL_QUALIFICATION')]:
        file=root/native/'completion-verification.json'
        if json.loads(file.read_text())['status']!=status:raise ValueError('Actual-shape full numerical proof required')
        numerical[native]=file_sha256(file)
    initial=root.parent/'recurrence/scale-tune-original-initializers-v1';initial_report=json.loads((initial/'report.json').read_text())
    if initial_report['phase']!='tune' or initial_report['seed']!=38213 or initial_report['optimizer_updates']!=0:
        raise ValueError('Original random TUNE initializers required')
    data=root/'natural-append-capacity-v4';packing=json.loads((data/'packed/manifest.json').read_text())
    counts=np.load(data/'packed/prefix_counts.npy',mmap_mode='r')
    def targets(update):return sum(map(int,counts[update*512]))
    inventory=profiled_inventory('medium');cases={};phases={};order=[];names=[];phase_limits={}
    for index,rate in enumerate([.008,.016,.032,.064]):
        for row in inventory:
            role=row['role'];key=role.replace('_','-')+'-lr'+str(index);record=next(r for r in initial_report['records'] if r['role']==role)
            if file_sha256(initial/record['file'])!=record['sha256']:raise AssertionError('Original TUNE C0 changed')
            artifact=IDENTITY+'/tune-artifacts/'+key;warm=IDENTITY+'-'+key+'-warm';replay=IDENTITY+'-'+key+'-replay'
            case=dict(phase='tune',seed=38213,state_seed=38246 if role=='fixed_two' else None,model_role=role,
                model=row['config'],parameters=row['parameters'],initializer='../recurrence/scale-tune-original-initializers-v1/'+record['file'],
                initializer_sha256=record['sha256'],packing_fingerprint=packing['fingerprint'],peak_lr=rate,lineage_case=key,
                endpoint_updates=300,stop_update=300,warmup_targets=targets(20),cooldown_targets=targets(270),end_targets=targets(300),
                work_per_update=fixed_update_work(row['config'],role)['total_per_update'],
                save_updates=[1,299,300] if index==0 else [299,300],capture_before_updates=[0,299] if index==0 else [299],
                audit_first_update=index==0,artifact_directory=artifact+'/warm',replay_directory=artifact+'/replay',
                audit_directory=IDENTITY+'/audits/'+key,score_directory=IDENTITY+'/scores/'+key,cohort='tune',
                retained_model='../recurrence/scale-medium-tune-retained-v1/'+key+'.pt',cleanup_receipt=IDENTITY+'/cleanup/'+key+'.json')
            cases[key]=case;replay_key=key+'-replay'
            cases[replay_key]={**case,'artifact_directory':case['replay_directory'],
                'resume':dict(directory=case['artifact_directory'],update=299),'save_updates':[300],
                'capture_before_updates':[], 'audit_first_update':False}
            phases[warm]=dict(case=key);phases[replay]=dict(case=replay_key)
            names.extend([warm,replay]);order.append(key);phase_limits[warm]=1500;phase_limits[replay]=600
    start=time.time();deadline=min(start+16200,SCALE_END)
    # Main endpoints are fixed now, but admission follows actual TUNE costs and
    # healthy independent rates. No changing them after seeing a main score.
    main_projection=14.1*3600
    if start+16200+main_projection>SCALE_END:raise TimeoutError('Complete tuning plus conservative main allowance does not fit original24h cap')
    mem=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if mem<(512+192)*1024**3:raise RuntimeError('Full192GiB Native allowance and protected512GiB required')
    if shutil.disk_usage(root).free<(35+32)*1024**3:raise RuntimeError('Complete32GiB transient TUNE allowance plus35GiB protected reserve required')
    remaining_model_bytes=sum(cases[k]['parameters']*4+256*1024**2 for k in order)
    if shutil.disk_usage(initial.parent).free<35*1024**3+remaining_model_bytes:
        raise RuntimeError('All eight retained TUNE prediction models must fit prospectively')
    if min(shutil.disk_usage('/').free,shutil.disk_usage(root.parent).free)<36*1024**3:raise RuntimeError('Protected root/output reserves')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for file in ['docs/SCALE_JOINT_DEVELOPMENT_20261027.md','tests/test_scale_training_state.py','tests/test_scale_training_fixture.py',
                 'tests/test_scale_work_ledger.py','scripts/run_native_isolated.sh']:sources[file]=file_sha256(file)
    worker,host=split_snapshot_sources(sources)
    plan=dict(identity=IDENTITY,purpose='scale_registered_development_v1',registered_unix=start,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        world=64,worker_memory_gib=192,seconds_limit=16200,phase_seconds_limit=1500,
        native_phase_seconds_limits=phase_limits,cpu_phase_seconds_limit=1800,readout_seconds_limit=900,
        absolute_deadline_unix=deadline,scale_budget_started_unix=SCALE_START,scale_budget_deadline_unix=SCALE_END,
        trainer_gate_receipt='scale-training-state-gate-v1/report.json',trainer_gate_sha256=file_sha256(gate),
        numerical_gates=numerical,source_audit_sha256=file_sha256(source_audit),
        initializer_report_sha256=file_sha256(initial/'report.json'),
        cases=cases,case_order=order,phases=phases,case_native_phases={k:[IDENTITY+'-'+k+'-warm',IDENTITY+'-'+k+'-replay'] for k in order},
        root_comm_ports=reserve_ports(root,names),
        data=dict(packed='natural-append-capacity-v4/packed',source_sha256=packing['source_sha256'],
            fingerprint=packing['fingerprint'],manifest_sha256=file_sha256(data/'packed/manifest.json')),
        evaluation=dict(corpus='natural-append-capacity-v4/dev.json',corpus_sha256=file_sha256(data/'dev.json'),
            scope='consumed_development_only',tune_documents_per_domain=64,selection='First64 documents per domain sorted by immutable ID in existing tune cohort; no model score used'),
        selection=dict(metric='weighted_nll',domain_weights=[.6,.2,.2],healthy_maximum_nll=4.,independent_per_role=True,
            tie_break='lowest rate',boundary_optimum='report unbracketed; no automatic grid expansion'),
        storage_lifecycle='successful_tune_diagnostics_after_audit_score_and_exact_final_weights',
        declared_retention='All C0, final FP32 prediction weights, failures, source, per-update/numerical/quality/cost/closure/hash receipts retained. Only listed successful TUNE captures released after complete gates; no raw source or main state release.',
        projected_main=dict(seed=49177,recurrent_updates=9554,recurrent_targets=5000260971,
            ordinary_work_updates=12939,ordinary_work_targets=6771820945,equal_data_updates=9554,
            cooldown_fraction=.1,warmup_targets=100000000,minimum_useful_gain=.01,
            main_admission='Separate frozen registration after healthy independently tuned roles, complete TUNE closure and a refreshed measured budget that fits the unchanged original24h cap. No automatic main or confirmation.'),
        actual_tune_update_allowance=2408,terminal_alpha_spent=0,scientific_main_fit_admitted=False,
        scope=Path('docs/SCALE_JOINT_DEVELOPMENT_20261027.md').read_text())
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_TUNE_NOT_ACTIVATED',scientific_main_fit_admitted=False))


def run(root):
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');records={};result=dict(status='STOPPED_SCALE_TUNE',records=records,scientific_main_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('Bounded TUNE interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for key in execution.plan['case_order']:
            case=execution.plan['cases'][key];record={};records[key]=record
            for suffix,case_key in [('warm',key),('replay',key+'-replay')]:
                name=IDENTITY+'-'+key+'-'+suffix
                options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                    '--plan-sha',execution.sha,'--case',case_key,'--comm-port',str(execution.plan['root_comm_ports'][name])]
                record[suffix]=execution.native(name,'scripts.train_scale_registered',options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE')
                write_state(folder/'progress.json',result)
            record['audit']=execution.cpu(key+'-audit','scripts.audit_scale_registered',
                ['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['audit_directory'])],
                root/case['audit_directory']/'report.json','PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
            record['score']=execution.cpu(key+'-score','scripts.score_scale_registered',
                ['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['score_directory'])],
                root/case['score_directory']/'report.json','COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
            execution.check()
            cleanup=subprocess.run(['sudo','-n','env','USE_TORCH_XLA=0','PYTHONDONTWRITEBYTECODE=1',
                'PYTHONPATH='+str(Path('.').resolve()),os.sys.executable,'-B','-m','scripts.close_scale_tune_diagnostics',
                '--root',str(root),'--plan',str(folder/'plan.json'),'--case',key],capture_output=True,text=True,timeout=300)
            if cleanup.returncode:raise RuntimeError('Preserved successful TUNE diagnostic lifecycle failed: '+cleanup.stderr[-2000:])
            record['cleanup']=json.loads((root/case['cleanup_receipt']).read_text());write_state(folder/'progress.json',result)
        selected={}
        for role in ['fixed_two','ordinary8']:
            keys=[k for k in execution.plan['case_order'] if execution.plan['cases'][k]['model_role']==role]
            eligible=[k for k in keys if records[k]['score']['metrics']['weighted_nll']<execution.plan['selection']['healthy_maximum_nll']]
            if not eligible:raise RuntimeError('No numerically healthy learning candidate for '+role)
            key=min(eligible,key=lambda k:(records[k]['score']['metrics']['weighted_nll'],execution.plan['cases'][k]['peak_lr']))
            selected[role]=dict(case=key,rate=execution.plan['cases'][key]['peak_lr'],score=records[key]['score']['metrics']['weighted_nll'],
                boundary_optimum=execution.plan['cases'][key]['peak_lr'] in [.008,.064])
        result.update(status='COMPLETE_INDEPENDENT_SCALE_RATE_SELECTION',selected=selected,
            scope='Development TUNE selection only. Main endpoints remain conditional on a separate frozen measured budget; no automatic larger run or terminal claim.')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(plan_sha256=execution.sha);execution.close(folder/'execution-costs.json')
        write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],scientific_main_fit_admitted=False))
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
