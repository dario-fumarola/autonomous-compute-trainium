"""Finish paired cheap-control branches;no terminal-confirmation allocation."""
import argparse
import concurrent.futures as cf
import json
import signal
import time
from pathlib import Path

from compute_trainium.future_span_execution import Execution
from compute_trainium.metrics import DocumentScore,paired_group_interval
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.register_mixing_control_main import register,IDENTITY,PUBLIC


def checkpoint(row):return next(r for r in row['checkpoint_records'] if r['update']==row['update'])


def run(root,plan,*,resume=False):
    out=root/IDENTITY;file=out/'plan.json'
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal MAIN')
    for relative,key in [('small-core-mixing-tune-v2/report.json','tune_report_sha256'),
        ('future-span-main-v2/report.json','original_main_report_sha256'),
        ('future-span-work-control-v1/report.json','original_work_report_sha256')]:
        if file_sha256(root/relative)!=plan[key]:raise AssertionError('Original completed evidence changed')
    execution=Execution(root,file,out/'execution',resume=resume);pool=cf.ThreadPoolExecutor(3)
    fits=[];gradients=[];scores=[];result=dict(status='STOPPED_CHEAP_MIXING_MAIN')
    def interrupted(signum,frame):execution.failed.set();raise KeyboardInterrupt('Cheap-control MAIN signal '+str(signum))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}

    def fit(key,phase,end,parent=None):
        name=IDENTITY+'-'+key+'-'+phase;case=plan['cases'][key]
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
            '--plan-sha',execution.sha,'--case',key,'--segment-end',str(end),
            '--comm-port',str(plan['root_comm_ports'][name])]
        if parent:
            saved=checkpoint(parent[1]);options+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{saved["update"]}.pt',
                                                 '--resume-sha',saved['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_mixing_control_scientific_v2',options,
            'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        count=end-(parent[1]['update'] if parent else 0)
        if (row['plan_sha256']!=execution.sha or row['case_id']!=key or row['update']!=end
                or row['qualification_only'] or row['actual_optimizer_calls_per_replica']!=count):
            raise AssertionError('Actual scientific control updates differ')
        if parent and (not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact']
                or row['recovered_state_digest']!=checkpoint(parent[1])['full_state_digest']):
            raise AssertionError('Full model/optimizer/RNG branch recovery differs')
        for saved in row['checkpoint_records']:
            if (not saved['all_complete_replica_states_equal'] or file_sha256(root/'runs'/name/f'checkpoint-u{saved["update"]}.pt')!=saved['checkpoint_sha256']):
                raise AssertionError('Immutable checkpoint or replica equality differs')
        if end==case['total_updates'] and (row['real_targets']!=case['end_targets']
                or row['valid_anchors']!=case['end_anchors'] or row['estimated_dense_matrix_ops']!=case['training_dense_matrix_ops']):
            raise AssertionError('Complete independent source/anchor/work endpoint differs')
        with execution.lock:
            fits.append(dict(identity=name,model_variant=plan['model_variant'],
                **{k:v for k,v in row.items() if k not in ['progress','first_global_probe']}))
            write_state(PUBLIC/'training-progress.json',dict(records=fits))
        return name,row

    def audit(key,capture,point):
        name,row=capture;saved=checkpoint(row);target=out/'gradients'/f'{key}-u{row["update"]}'
        checked=execution.cpu(key+f'-u{row["update"]}-gradient','scripts.audit_mixing_scientific_gradient',
            ['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,'--case',key,
             '--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),'--checkpoint-sha',saved['checkpoint_sha256'],
             '--capture',str(root/'runs'/point),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200'],
            target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        if checked['checkpoint_sha256']!=saved['checkpoint_sha256'] or checked['training_update']!=row['update']:
            raise AssertionError('Independent scientific gradient covers another endpoint')
        return dict(case_id=key,update=row['update'],native_identity=point,independent_cpu=checked)

    def native_job(key,phase,end,parent=None):
        try:
            capture=fit(key,phase,end,parent);name,row=capture;saved=checkpoint(row)
            point=IDENTITY+'-'+key+('-preflight-gradient' if phase=='preflight' else '-final-gradient')
            observed=execution.native(point,'scripts.capture_mixing_control_gradient',
                ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                 '--plan-sha',execution.sha,'--case',key,'--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{end}.pt',
                 '--checkpoint-sha',saved['checkpoint_sha256'],'--comm-port',str(plan['root_comm_ports'][point])],
                'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
            if observed['checkpoint_sha256']!=saved['checkpoint_sha256'] or observed['optimizer_updates']!=0:
                raise AssertionError('Zero-update numerical capture differs')
            return capture,pool.submit(audit,key,capture,point)
        except BaseException:execution.failed.set();raise

    def score(key,capture,depth):
        name,row=capture;saved=checkpoint(row);target=out/'scores'/f'{key}-d{depth}'
        checked=execution.cpu(key+f'-score-d{depth}','scripts.evaluate_mixing_control_cpu',
            ['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,
             '--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),'--checkpoint-sha',saved['checkpoint_sha256'],
             '--case',key,'--cohort','curve','--depth',str(depth),'--output',str(target),
             '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200'],
            target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
        if (checked['checkpoint_sha256']!=saved['checkpoint_sha256'] or checked['training_real_targets']!=plan['cases'][key]['end_targets']
                or not checked['head_removed'] or not checked['full_state_readback_exact']):
            raise AssertionError('Independent cooled cheap-control readout differs')
        return {**checked,'model_variant':plan['model_variant']}

    try:
        short=[k for k in plan['cases'] if k.endswith('equal-data')];parents={};stable={}
        with cf.ThreadPoolExecutor(2) as native:
            jobs={k:native.submit(native_job,k,'preflight',2) for k in short}
            for key,future in jobs.items():
                capture,gate=future.result();gradients.append(gate.result(timeout=max(1,execution.deadline-time.time())));parents[key]=capture
        write_state(PUBLIC/'integration-pass.json',dict(status='PASS_BOTH_FRESH_SEED_COUNTED_INTEGRATIONS',
            updates_per_seed=2,included_in_terminal_endpoints=True))
        with cf.ThreadPoolExecutor(2) as native:
            jobs={k:native.submit(fit,k,'stable',960,parents[k]) for k in short}
            for key,future in jobs.items():stable[plan['cases'][key]['seed']]=future.result()
        handles=[]
        with cf.ThreadPoolExecutor(2) as native:
            jobs={k:native.submit(native_job,k,'endpoint',case['total_updates'],stable[case['seed']])
                  for k,case in plan['cases'].items()}
            for key,future in jobs.items():
                capture,gate=future.result();gradients.append(gate.result(timeout=max(1,execution.deadline-time.time())))
                handles.append(pool.submit(score,key,capture,2))
                if key.endswith('equal-data'):handles.append(pool.submit(score,key,capture,1))
        for handle in handles:scores.append(handle.result(timeout=max(1,execution.deadline-time.time())))
        if len(scores)!=6 or len(gradients)!=6 or len({(s['case_id'],s['depth']) for s in scores})!=6:
            raise AssertionError('All endpoints,diagnostic exits and numerical gates required')
        def documents(p):
            return [DocumentScore(r['domain'],r['id'],r['group'],r['nll_sum'],r['real_targets'])
                    for r in json.loads(p.read_text())['records']]
        contrasts={}
        for seed in [5231,7283]:
            path=root/'future-span-main-v2/scores'/f'recurrent-ntp-s{seed}-lr32000-cooled1b-d2/document-scores.json'
            recurrent=documents(path);contrasts[seed]={}
            for endpoint in ['equal-data','matched-work']:
                key=f'recurrent-ntp-s{seed}-{endpoint}'
                control=documents(out/'scores'/f'{key}-d2/document-scores.json')
                contrasts[seed][endpoint]=paired_group_interval(control,recurrent,draws=2000,seed=31579+seed)
        execution.check();result=dict(status='COMPLETE_BOUNDED_CHEAP_MIXING_MAIN',scores=scores,
            gradient_gates=gradients,recurrent_gain_over_cheap_control=contrasts)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),scores=scores,gradient_gates=gradients,training_records=fits);raise
    finally:
        execution.failed.set();pool.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for s,h in handlers.items():signal.signal(s,h)
        result.update(plan_sha256=execution.sha,model_variant=plan['model_variant'],test_accessed=False,
            terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();file=args.root/IDENTITY/'plan.json'
    plan=json.loads(file.read_text()) if file.exists() else register(args.root)
    if not args.prepare_only:run(args.root,plan,resume=args.resume)


if __name__=='__main__':main()
