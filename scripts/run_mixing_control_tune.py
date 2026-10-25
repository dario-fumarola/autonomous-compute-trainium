"""Three fair LR trials,with counted scientific-driver integration gates."""
import argparse
import concurrent.futures as cf
import json
import signal
import time
from pathlib import Path

from compute_trainium.future_span_execution import Execution
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.register_mixing_control_tune import register,IDENTITY,PUBLIC


def checkpoint(report):
    return next(r for r in report['checkpoint_records'] if r['update']==report['update'])


def run(root,plan,*,resume=False):
    out=root/IDENTITY;file=out/'plan.json'
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal tuning')
    hardware=root/'small-core-mixing-qualification-v1/report.json'
    if file_sha256(hardware)!=plan['hardware_report_sha256']:
        raise AssertionError('Complete original hardware pass changed')
    execution=Execution(root,file,out/'execution',resume=resume)
    pool=cf.ThreadPoolExecutor(3);fits=[];gradients=[];scores=[]
    result=dict(status='STOPPED_CHEAP_MIXING_TUNE')
    def interrupted(signum,frame):
        execution.failed.set();raise KeyboardInterrupt('Mixing TUNE signal '+str(signum))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}

    def fit(key,phase,parent=None):
        case=plan['cases'][key];end=2 if phase=='preflight' else 576
        name=IDENTITY+'-'+key+'-'+phase
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
            '--plan-sha',execution.sha,'--case',key,'--segment-end',str(end),
            '--comm-port',str(plan['root_comm_ports'][name])]
        if parent:
            row=checkpoint(parent[1]);options += ['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{row["update"]}.pt',
                                                '--resume-sha',row['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_mixing_control_scientific',options,
            'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        if (row['plan_sha256']!=execution.sha or row['case_id']!=key or row['update']!=end or row['qualification_only']
                or row['actual_optimizer_calls_per_replica']!=end-(2 if parent else 0)):
            raise AssertionError('Actual counted scientific updates differ')
        if parent and (not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact']
                or row['recovered_state_digest']!=checkpoint(parent[1])['full_state_digest']):
            raise AssertionError('Scientific driver full-state/RNG recovery differs')
        for saved in row['checkpoint_records']:
            if (not saved['all_complete_replica_states_equal']
                    or file_sha256(root/'runs'/name/f'checkpoint-u{saved["update"]}.pt')!=saved['checkpoint_sha256']):
                raise AssertionError('Scientific checkpoint/replicas differ')
        if end==576 and (row['real_targets']!=case['end_targets']
                or row['estimated_dense_matrix_ops']!=case['training_dense_matrix_ops']
                or row['valid_anchors']!=case['end_anchors']):
            raise AssertionError('Scientific terminal source/anchor/work counters differ')
        with execution.lock:
            fits.append(dict(identity=name,model_variant=plan['model_variant'],
                **{k:v for k,v in row.items() if k not in ['progress','first_global_probe']}))
            write_state(PUBLIC/'training-progress.json',dict(records=fits))
        return name,row

    def audit(key,capture,point):
        name,row=capture;saved=checkpoint(row)
        target=out/'gradients'/f'{key}-u{row["update"]}'
        scored=execution.cpu(key+f'-u{row["update"]}-gradient','scripts.audit_mixing_scientific_gradient',
            ['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,'--case',key,
             '--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),
             '--checkpoint-sha',saved['checkpoint_sha256'],'--capture',str(root/'runs'/point),
             '--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200'],
             target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        if scored['checkpoint_sha256']!=saved['checkpoint_sha256'] or scored['training_update']!=row['update']:
            raise AssertionError('Independent full-gradient scientific point differs')
        return dict(case_id=key,update=row['update'],native_identity=point,independent_cpu=scored)

    def native_job(key,phase,parent=None):
        try:
            capture=fit(key,phase,parent);name,row=capture;saved=checkpoint(row)
            point=IDENTITY+'-'+key+('-preflight-gradient' if phase=='preflight' else '-final-gradient')
            observed=execution.native(point,'scripts.capture_mixing_control_gradient',
                ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                 '--plan-sha',execution.sha,'--case',key,'--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{row["update"]}.pt',
                 '--checkpoint-sha',saved['checkpoint_sha256'],'--comm-port',str(plan['root_comm_ports'][point])],
                'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
            if observed['checkpoint_sha256']!=saved['checkpoint_sha256'] or observed['optimizer_updates']!=0:
                raise AssertionError('Native scientific gradient point differs')
            return capture,pool.submit(audit,key,capture,point)
        except BaseException:
            execution.failed.set();raise

    def score(key,capture):
        name,row=capture;saved=checkpoint(row);target=out/'scores'/key
        result=execution.cpu(key+'-tune-score','scripts.evaluate_mixing_control_cpu',
            ['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,
             '--checkpoint',str(root/'runs'/name+'/checkpoint-u576.pt'),'--checkpoint-sha',saved['checkpoint_sha256'],
             '--case',key,'--cohort','tune','--depth','2','--output',str(target),
             '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200'],
            target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
        if (result['checkpoint_sha256']!=saved['checkpoint_sha256'] or result['training_update']!=576
                or result['training_real_targets']!=plan['cases'][key]['end_targets']):
            raise AssertionError('Cooled control tuning endpoint differs')
        return {**result,'model_variant':plan['model_variant'],'peak_lr':plan['cases'][key]['peak_lr']}
    try:
        parents={}
        with cf.ThreadPoolExecutor(2) as native_pool:
            jobs={k:native_pool.submit(native_job,k,'preflight') for k in plan['cases']}
            for key,future in jobs.items():
                capture,gate=future.result();gradients.append(gate.result(timeout=max(1,execution.deadline-time.time())))
                parents[key]=capture
        if len(gradients)!=3:raise AssertionError('All three integration gradients precede longer training')
        write_state(PUBLIC/'integration-pass.json',dict(status='PASS_ALL_THREE_COUNTED_SCIENTIFIC_INTEGRATIONS',
            research_updates_per_case=2,included_in_terminal_endpoint=True))
        with cf.ThreadPoolExecutor(2) as native_pool:
            jobs={k:native_pool.submit(native_job,k,'finish',parents[k]) for k in plan['cases']}
            handles=[]
            for key,future in jobs.items():
                capture,gate=future.result();gradients.append(gate.result(timeout=max(1,execution.deadline-time.time())))
                handles.append(pool.submit(score,key,capture))
            for future in handles:scores.append(future.result(timeout=max(1,execution.deadline-time.time())))
        if len(scores)!=3 or len(gradients)!=6 or {s['case_id'] for s in scores}!=set(plan['cases']):
            raise AssertionError('All unique trials and complete numerical gates required')
        winner=min(scores,key=lambda s:(s['metrics']['weighted_nll'],s['peak_lr']))
        execution.check()
        result=dict(status='COMPLETE_BOUNDED_CHEAP_MIXING_TUNE',scores=scores,gradient_gates=gradients,
            selected_lr=winner['peak_lr'],selected_tune_nll=winner['metrics']['weighted_nll'],
            selected_at_grid_edge=winner['peak_lr'] in [.008,.032],scientific_main_admitted=False)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),scores=scores,gradient_gates=gradients,training_records=fits);raise
    finally:
        execution.failed.set();pool.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for s,h in handlers.items():signal.signal(s,h)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False,
            model_variant=plan['model_variant'],scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();file=args.root/IDENTITY/'plan.json'
    plan=json.loads(file.read_text()) if file.exists() else register(args.root)
    if not args.prepare_only:run(args.root,plan,resume=args.resume)


if __name__=='__main__':main()
