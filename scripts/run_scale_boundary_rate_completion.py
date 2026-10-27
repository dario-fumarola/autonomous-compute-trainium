"""Complete one frozen boundary-corrected rate grid; never launches main."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-boundary-rate-completion-v1'
PUBLIC=Path('results/scale_boundary_rate_completion_20261027')
REUSE={
 'ordinary8-lr0':('scale-medium-tune-65536-v1/plan.json','scale-tune-readout-recovery-v1/progress.json','records'),
 'ordinary8-lr1':('scale-ordinary-lr1-continuation-v1/plan.json','scale-ordinary-lr1-continuation-v1/report.json','record'),
 'fixed-two-lr1':('scale-boundary-learning-lr1-v1/plan.json','scale-boundary-learning-lr1-v1/report.json','record')}


def eligible_reuse(root,key,entry,parent):
    plan_path,result_path,kind=entry;plan=json.loads((root/plan_path).read_text());result=json.loads((root/result_path).read_text())
    record=result[kind][key] if kind=='records' else result[kind];case=plan['cases'][key];sha=file_sha256(root/plan_path)
    audit,score=record['audit'],record['score']
    for field in ['model','model_role','initializer_sha256','peak_lr','seed','state_seed','endpoint_updates','end_targets','warmup_targets','cooldown_targets']:
        if case[field]!=parent['cases'][key][field]:raise AssertionError('Reuse differs from original fixed opportunity:'+field)
    if (plan['data']!=parent['data'] or plan['evaluation']!=parent['evaluation'] or plan.get('manual_loss_scale')!=65536
            or audit['study_sha256']!=sha or score['study_sha256']!=sha
            or audit['status']!='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT'
            or not audit['full512']['gradient']['passes_two_percent_gate'] or abs(audit['full512']['native_minus_cpu_nll'])>.003
            or audit['full512']['windows']!=512 or not audit['replay']['passes']
            or score['status']!='COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT' or not score['model_unchanged']
            or score['documents']!=192 or not record['warm']['cooldown_complete']):raise ValueError('Only completely audited,identical eligible opportunities may be reused')
    if file_sha256(root/case['initializer'])!=case['initializer_sha256'] or file_sha256(root/case['retained_model'])!=score['retained_model_sha256']:
        raise AssertionError('Exact original C0 or retained prediction bytes changed')
    for name in plan['case_native_phases'][key]:
        closure=root/'snapshots'/name/'closure.json';native=root/'runs'/name/'report.json'
        if json.loads(closure.read_text())['exit_code']!=0 or json.loads(native.read_text())['plan_sha256']!=sha:raise AssertionError('Successful exact original Native closure required')
    if key=='fixed-two-lr1' and result['status']!='COMPLETE_FRESH_BOUNDARY_CORRECTED_LEARNING_CELL':raise ValueError('Current corrected cell must finish before any completion stage')
    return dict(origin='reused_completed_fixed_opportunity',source_plan=plan_path,source_plan_sha256=sha,source_result=result_path,
        source_result_sha256=file_sha256(root/result_path),retained_model=case['retained_model'],audit=audit,score=score,
        added_optimizer_updates=0,added_assigned_chip_hours=0)


def prepare(root):
    if managed_running():raise RuntimeError('Complete previous Native closure required')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every fixed-grid attempt')
    scope_path=root/'scale-boundary-scope-freeze-v1/scope.json';scope=json.loads(scope_path.read_text())
    if file_sha256('docs/SCALE_BOUNDARY_RATE_COMPLETION_20261027.md')!=scope['scope_sha256']:raise AssertionError('Frozen scope document changed')
    if time.time()+9000>scope['rate_completion_deadline_unix']:raise TimeoutError('Measured remaining150-minute allowance must fit frozen14:45UTC cutoff')
    proof=root/'scale-boundary-fixed-two-numerical-v1/completion-verification.json'
    if file_sha256(proof)!=scope['boundary_qualification_sha256'] or json.loads(proof.read_text())['status']!='PASS_BOUNDARY_COMPLETE_NUMERICAL_QUALIFICATION':raise ValueError('Frozen complete boundary proof required')
    ordinary_proof=root/'scale-65536-ordinary-numerical-v1/completion-verification.json'
    if json.loads(ordinary_proof.read_text())['status']!='PASS_SCALE_65536_COMPLETE_NUMERICAL_QUALIFICATION':raise ValueError('Unchanged ordinary numerical proof required')
    parent_path=root/'scale-medium-tune-65536-v1/plan.json';parent=json.loads(parent_path.read_text());reused={k:eligible_reuse(root,k,v,parent) for k,v in REUSE.items()}
    if shutil.disk_usage(root).free<53*1024**3:raise RuntimeError('Complete18GiB warm/replay/failure allowance plus35GiB required')
    retention=sum(parent['cases'][k]['parameters']*4+256*1024**2 for k in scope['fresh_remaining_cases'])
    if shutil.disk_usage(root.parent/'recurrence').free<35*1024**3+retention:raise RuntimeError('Every remaining retained prediction model plus reserve required')
    cases={};phases={};names=[];limits={}
    for key in scope['fresh_remaining_cases']:
        case=dict(parent['cases'][key]);artifact=IDENTITY+'/artifacts/'+key
        case.update(artifact_directory=artifact+'/warm',replay_directory=artifact+'/replay',audit_directory=IDENTITY+'/audits/'+key,
            score_directory=IDENTITY+'/scores/'+key,retained_model='../recurrence/'+IDENTITY+'/retained/'+key+'.pt',cleanup_receipt=IDENTITY+'/cleanup/'+key+'.json')
        cases[key]=case;cases[key+'-replay']={**case,'artifact_directory':case['replay_directory'],
            'resume':dict(directory=case['artifact_directory'],update=299),'save_updates':[300],'capture_before_updates':[],'audit_first_update':False}
        for suffix,ck,seconds in [('warm',key,1500),('replay',key+'-replay',600)]:
            name=IDENTITY+'-'+key+'-'+suffix;names.append(name);phases[name]=dict(case=ck);limits[name]=seconds
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for name in ['scripts/run_native_isolated.sh','docs/SCALE_BOUNDARY_RATE_COMPLETION_20261027.md','tests/test_scale_precision_probe.py']:sources[name]=file_sha256(name)
    qualified=json.loads((proof.parent/'plan.json').read_text())
    for name,sha in qualified['source_sha256'].items():
        if (name.startswith('compute_trainium/') or name in ['scripts/train_scale_registered.py','scripts/train_scale_capture_invalid_gradient.py','scripts/audit_scale_registered.py']) and file_sha256(name)!=sha:raise AssertionError('Qualified mathematical/training source changed:'+name)
    worker,host=split_snapshot_sources(sources);start=time.time()
    plan={**parent,'identity':IDENTITY,'registered_unix':start,'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host,
        'precision_policy':'fp32_recurrent_boundaries','manual_loss_scale':65536,'invalid_gradient_capture':True,
        'seconds_limit':scope['rate_completion_deadline_unix']-start,'absolute_deadline_unix':scope['rate_completion_deadline_unix'],
        'original_incomplete_tune_clock_unchanged':True,'scope_freeze_sha256':file_sha256(scope_path),'numerical_gates':{str(proof.relative_to(root)):file_sha256(proof),str(ordinary_proof.relative_to(root)):file_sha256(ordinary_proof)},
        'cases':cases,'case_order':scope['fresh_remaining_cases'],'selection_case_order':parent['case_order'],'reused_cases':reused,
        'phases':phases,'native_phase_seconds_limits':limits,'root_comm_ports':reserve_ports(root,names),
        'case_native_phases':{k:[IDENTITY+'-'+k+'-'+s for s in ['warm','replay']] for k in scope['fresh_remaining_cases']},
        'allowed_tune_artifact_roots':[IDENTITY+'/artifacts'],'actual_new_tune_update_allowance':5*301,
        'scope':scope['scope'],'scientific_main_fit_admitted':False}
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');records=dict(execution.plan['reused_cases'])
    result=dict(status='STOPPED_BOUNDARY_RATE_COMPLETION',records=records,scientific_main_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('Frozen rate-completion stage stopped')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for key in execution.plan['case_order']:
            case=execution.plan['cases'][key];record={};records[key]=record
            for suffix,ck in [('warm',key),('replay',key+'-replay')]:
                name=IDENTITY+'-'+key+'-'+suffix;options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',ck,'--comm-port',str(execution.plan['root_comm_ports'][name])]
                record[suffix]=execution.native(name,'scripts.train_scale_capture_invalid_gradient',options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE');write_state(folder/'progress.json',result)
            record['audit']=execution.cpu(key+'-audit','scripts.audit_scale_registered',['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['audit_directory'])],root/case['audit_directory']/'report.json','PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
            record['score']=execution.cpu(key+'-score','scripts.score_scale_registered',['--root',str(root),'--plan',str(folder/'plan.json'),'--case',key,'--output',str(root/case['score_directory'])],root/case['score_directory']/'report.json','COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
            cleanup=subprocess.run(['sudo','-n','env','USE_TORCH_XLA=0','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(Path('.').resolve()),os.sys.executable,'-B','-m','scripts.close_scale_tune_diagnostics','--root',str(root),'--plan',str(folder/'plan.json'),'--case',key],capture_output=True,text=True,timeout=300)
            if cleanup.returncode:raise RuntimeError('Successful-only diagnostic closure failed:'+cleanup.stderr[-1000:])
            record['cleanup_receipt_sha256']=file_sha256(root/case['cleanup_receipt']);write_state(folder/'progress.json',result)
        selected={}
        for role in ['fixed_two','ordinary8']:
            keys=[k for k in execution.plan['selection_case_order'] if k.startswith('fixed-two' if role=='fixed_two' else 'ordinary8')]
            eligible=[k for k in keys if records[k]['score']['metrics']['weighted_nll']<4.]
            if not eligible:raise RuntimeError('No eligible independently tuned role:'+role)
            rate=lambda k:execution.plan['cases'][k]['peak_lr'] if k in execution.plan['cases'] else [.008,.016,.032,.064][int(k[-1])]
            key=min(eligible,key=lambda k:(records[k]['score']['metrics']['weighted_nll'],rate(k)))
            selected[role]=dict(case=key,rate=rate(key),score=records[key]['score']['metrics']['weighted_nll'],boundary_optimum=rate(key) in [.008,.064])
        result.update(status='COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION',selected=selected)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(folder/'execution-costs.json');result['plan_sha256']=execution.sha;write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
