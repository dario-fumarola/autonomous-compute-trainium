"""One fresh boundary-corrected0.016 cell; old failed history remains excluded."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-boundary-learning-lr1-v1'
PUBLIC=Path('results/scale_boundary_learning_cell_20261027')
KEY='fixed-two-lr1'


def prepare(root):
    if managed_running():raise RuntimeError('Qualification must be terminal before admission')
    proof_file=root/'scale-boundary-fixed-two-numerical-v1/completion-verification.json'
    proof=json.loads(proof_file.read_text())
    if (proof['status']!='PASS_BOUNDARY_COMPLETE_NUMERICAL_QUALIFICATION'
            or not proof['first_update_moments_pass'] or not proof['full512']['gradient']['passes_two_percent_gate']
            or not proof['replay']['passes']):raise ValueError('Every fresh boundary qualification gate required')
    qualified=json.loads((proof_file.parent/'plan.json').read_text())
    for f,sha in qualified['source_sha256'].items():
        if (f.startswith('compute_trainium/') or f in ['scripts/train_scale_registered.py','scripts/train_scale_capture_invalid_gradient.py','scripts/audit_scale_registered.py']) and file_sha256(f)!=sha:
            raise AssertionError('Qualified mathematical/training source changed:'+f)
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve all corrected learning attempts')
    parent_file=root/'scale-medium-tune-65536-v1/plan.json';parent=json.loads(parent_file.read_text());start=time.time()
    if start+2100>parent['absolute_deadline_unix']:raise TimeoutError('Complete measured35-minute cell must fit original TUNE cutoff')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<704*1024**3 or shutil.disk_usage(root).free<49*1024**3:raise RuntimeError('192GiB worker and14GiB capture allowance plus protected reserves required')
    case=dict(parent['cases'][KEY]);case.update(artifact_directory=IDENTITY+'/artifacts/warm',replay_directory='../recurrence/'+IDENTITY+'/replay',
        audit_directory=IDENTITY+'/audit',score_directory=IDENTITY+'/score',retained_model='../recurrence/'+IDENTITY+'/retained-model.pt',cleanup_receipt=IDENTITY+'/cleanup.json')
    if shutil.disk_usage(root.parent/'recurrence').free<35*1024**3+6*1024**3:raise RuntimeError('Complete unshared replay and final prediction weight allowance required')
    if file_sha256(root/case['initializer'])!=case['initializer_sha256']:raise AssertionError('Original random TUNE C0 changed')
    replay={**case,'artifact_directory':case['replay_directory'],'resume':dict(directory=case['artifact_directory'],update=299),
        'save_updates':[300],'capture_before_updates':[],'audit_first_update':False}
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for f in ['scripts/run_native_isolated.sh','tests/test_scale_precision_probe.py']:sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-warm',IDENTITY+'-replay']
    plan={**parent,'identity':IDENTITY,'registered_unix':start,'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host,
        'precision_policy':'fp32_recurrent_boundaries','manual_loss_scale':65536,'invalid_gradient_capture':True,
        'seconds_limit':2100,'absolute_deadline_unix':start+2100,'unchanged_original_tune_deadline_unix':parent['absolute_deadline_unix'],
        'cases':{KEY:case,KEY+'-replay':replay},'case_order':[KEY],
        'phases':{names[0]:dict(case=KEY),names[1]:dict(case=KEY+'-replay')},
        'native_phase_seconds_limits':{names[0]:1500,names[1]:600},'root_comm_ports':reserve_ports(root,names),
        'allowed_tune_artifact_roots':[IDENTITY+'/artifacts','../recurrence/'+IDENTITY],
        'case_native_phases':{KEY:names},'boundary_complete_qualification_sha256':file_sha256(proof_file),
        'parent_original_plan_sha256':file_sha256(parent_file),'scientific_main_fit_admitted':False,
        'scope':'One fresh corrected-precision0.016 learning cell from the original zero-update38213 C0. Same architecture,authentic source prefix,state seed,objective,optimizer,schedule and300-update opportunity;FP32 recurrent boundaries now qualified,65536 scaler unchanged.35minutes total must fit the original11:58:26UTC cutoff and unchanged initial24h scale cap. Full mature512 audit,actual replay,complete cooldown and consumed192-document TUNE readout precede any accepted score. Old failed/reproduced histories remain ineligible and charged. No rate selection,complete grid,main fit,renewed clock or terminal claim follows this single cell.'}
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');record={}
    result=dict(status='STOPPED_BOUNDARY_CORRECTED_LEARNING_CELL',record=record,scientific_main_fit_admitted=False,rate_selected=False)
    def stop(number,frame):raise KeyboardInterrupt('Original bounded learning cell stopped')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        case=execution.plan['cases'][KEY]
        for suffix,ck in [('warm',KEY),('replay',KEY+'-replay')]:
            name=IDENTITY+'-'+suffix
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',ck,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            record[suffix]=execution.native(name,'scripts.train_scale_capture_invalid_gradient',options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE');write_state(folder/'progress.json',result)
        record['audit']=execution.cpu('audit','scripts.audit_scale_registered',['--root',str(root),'--plan',str(folder/'plan.json'),'--case',KEY,'--output',str(root/case['audit_directory'])],root/case['audit_directory']/'report.json','PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
        record['score']=execution.cpu('score','scripts.score_scale_registered',['--root',str(root),'--plan',str(folder/'plan.json'),'--case',KEY,'--output',str(root/case['score_directory'])],root/case['score_directory']/'report.json','COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
        cleanup=subprocess.run(['sudo','-n','env','USE_TORCH_XLA=0','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(Path('.').resolve()),os.sys.executable,'-B','-m','scripts.close_scale_tune_diagnostics','--root',str(root),'--plan',str(folder/'plan.json'),'--case',KEY],capture_output=True,text=True,timeout=300)
        if cleanup.returncode:raise RuntimeError('Successful-only diagnostic closure failed:'+cleanup.stderr[-1000:])
        record['cleanup']=json.loads((root/case['cleanup_receipt']).read_text());result['status']='COMPLETE_FRESH_BOUNDARY_CORRECTED_LEARNING_CELL'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(folder/'execution-costs.json');result['plan_sha256']=execution.sha;write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
