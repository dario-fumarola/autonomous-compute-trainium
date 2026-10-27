"""One unstarted, preregistered ordinary cell; no new hyperparameter opportunity."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path

from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-ordinary-lr1-continuation-v1';PUBLIC=Path('results/scale_ordinary_lr1_continuation_20261027')


def prepare(root):
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every cell continuation')
    original=root/'scale-medium-tune-65536-v1/plan.json';parent=json.loads(original.read_text());failed=root/'scale-tune-readout-recovery-v1/report.json'
    if json.loads(failed.read_text())['status']!='STOPPED_SCALE_READOUT_RECOVERY':raise ValueError('Preserved stopped parent required')
    if any(file_sha256(f)!=s for f,s in parent['source_sha256'].items()):raise AssertionError('Original scientific sources changed')
    if time.time()+1800>parent['absolute_deadline_unix']:raise TimeoutError('Complete30-minute cell must fit original TUNE cutoff')
    mem=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if mem<704*1024**3 or shutil.disk_usage(root).free<55*1024**3:raise RuntimeError('Complete192GiB worker and20GiB capture allowance plus protected reserves required')
    case=dict(parent['cases']['ordinary8-lr1']);case.update(artifact_directory=IDENTITY+'/artifacts/warm',replay_directory=IDENTITY+'/artifacts/replay',
        audit_directory=IDENTITY+'/audit',score_directory=IDENTITY+'/score',retained_model='../recurrence/scale-medium-tune-retained-v1/ordinary8-lr1.pt',cleanup_receipt=IDENTITY+'/cleanup.json')
    replay={**case,'artifact_directory':case['replay_directory'],'resume':dict(directory=case['artifact_directory'],update=299),
        'save_updates':[300],'capture_before_updates':[],'audit_first_update':False}
    cases={'ordinary8-lr1':case,'ordinary8-lr1-replay':replay};names=[IDENTITY+'-warm',IDENTITY+'-replay']
    sources=dict(parent['source_sha256'])
    for f in ['scripts/train_scale_capture_invalid_gradient.py','scripts/continue_scale_ordinary_lr1.py',
              'scripts/score_scale_readout_recovery.py','compute_trainium/canonical_prediction_checkpoint.py']:
        sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources)
    plan={**parent,'identity':IDENTITY,'registered_unix':time.time(),'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host,
        'seconds_limit':1800,'absolute_deadline_unix':min(time.time()+1800,parent['absolute_deadline_unix']),
        'invalid_gradient_capture':True,'cases':cases,'case_order':['ordinary8-lr1'],
        'phases':{names[0]:dict(case='ordinary8-lr1'),names[1]:dict(case='ordinary8-lr1-replay')},
        'native_phase_seconds_limits':{names[0]:1500,names[1]:600},'root_comm_ports':reserve_ports(root,names),
        'allowed_tune_artifact_roots':[IDENTITY+'/artifacts'],'case_native_phases':{'ordinary8-lr1':names},
        'original_registered_case':'ordinary8-lr1','original_plan_sha256':file_sha256(original),
        'parent_stopped_report_sha256':file_sha256(failed),'adds_hyperparameter_opportunities':False,
        'scope':parent['scope']+'\nOne unstarted original ordinary0.016 cell only. Same original zero-update C0,source,rate,schedule,precision,operators,normalization and300-update opportunity. The sole worker addition captures every raw invalid gradient/model/optimizer/RNG shard before rethrowing; it changes no successful update. Failed original recurrent0.016 remains ineligible; its failing state was not captured and is not retroactively reconstructed. No main selection or renewed original TUNE cutoff.'}
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    old_binding=json.loads((root/'scale-tune-readout-recovery-v1/plan.json').read_text())
    binding={**old_binding,'identity':IDENTITY+'-readout-only','original_plan_sha256':file_sha256(folder/'plan.json'),
        'replacement_retained_path':str(folder/'no-prior-failed-readout'),
        'absolute_deadline_unix':plan['absolute_deadline_unix']}
    write_state(folder/'readout-binding.json',binding)


def run(root):
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');record={};result=dict(status='STOPPED_REGISTERED_ORDINARY_CELL',record=record,scientific_main_fit_admitted=False)
    try:
        key='ordinary8-lr1';case=execution.plan['cases'][key]
        for suffix,ck in [('warm',key),('replay',key+'-replay')]:
            name=IDENTITY+'-'+suffix
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',ck,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            record[suffix]=execution.native(name,'scripts.train_scale_capture_invalid_gradient',options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE')
            write_state(folder/'progress.json',result)
        record['audit']=execution.cpu('audit','scripts.audit_scale_registered',['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['audit_directory'])],root/case['audit_directory']/'report.json','PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
        recovery=folder/'readout-binding.json'
        record['score']=execution.cpu('score','scripts.score_scale_readout_recovery',['--readout-recovery',str(recovery),'--readout-recovery-sha',file_sha256(recovery),'--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['score_directory'])],root/case['score_directory']/'report.json','COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
        cp=subprocess.run(['sudo','-n','env','USE_TORCH_XLA=0','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(Path('.').resolve()),os.sys.executable,'-B','-m','scripts.close_scale_tune_diagnostics','--root',str(root),'--plan',str(folder/'plan.json'),'--case',key],capture_output=True,text=True,timeout=300)
        if cp.returncode:raise RuntimeError('Prospectively declared successful-only cleanup failed: '+cp.stderr[-1000:])
        record['cleanup']=json.loads((root/case['cleanup_receipt']).read_text());result['status']='COMPLETE_ORIGINAL_REGISTERED_ORDINARY_LR1_CELL'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(folder/'execution-costs.json');result['plan_sha256']=execution.sha;write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
