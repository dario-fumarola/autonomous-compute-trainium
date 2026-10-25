"""Read-only score recovery after all original fits/gates;no clock extension."""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

from compute_trainium.future_span_execution import Execution
from compute_trainium.mixing_scoring_recovery import completed_fits,score_arguments
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='small-core-mixing-tune-scoring-v3'
PUBLIC=Path('results/small_core_mixing_tune_20261025/scoring-recovery-v3')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    root=args.root;source_file=root/'small-core-mixing-tune-v2/plan.json'
    source=json.loads(source_file.read_text());out=root/IDENTITY;file=out/'plan.json'
    if not file.exists():
        additions=['scripts/recover_mixing_control_scores.py','compute_trainium/mixing_scoring_recovery.py',
                   'tests/test_mixing_scoring_recovery.py']
        plan=dict(identity=IDENTITY,registered_unix=time.time(),source_tune_plan_sha256=file_sha256(source_file),
            source_sha256={**source['source_sha256'],**{p:file_sha256(p) for p in additions}},
            original_deadline_unix=source['original_absolute_deadline_unix'],
            seconds_limit=max(1,source['original_absolute_deadline_unix']-time.time()),
            cpu_phase_seconds_limit=1200,phase_seconds_limit=1,root_comm_ports={},
            scope='Score-only recovery for the proven Path+str launch error,after all three unchanged fits and all six independent512-window gates complete. Same three rates,cooled576-update checkpoints,consumed TUNE panel,depth2 and selection. No retraining,new endpoint,model change,held-out selection rule change,TEST or terminal alpha. Preserve the original16:50:26 UTC absolute cutoff and all prior costs.')
        out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
        write_state(file,plan);write_state(PUBLIC/'plan.json',plan)
    plan=json.loads(file.read_text())
    if args.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal score recovery')
    result=dict(status='STOPPED_CHEAP_MIXING_SCORE_RECOVERY');execution=None;scores=[]
    try:
        while True:
            if (time.time()>plan['original_deadline_unix']
                    or file_sha256(source_file)!=plan['source_tune_plan_sha256']
                    or any(file_sha256(p)!=h for p,h in plan['source_sha256'].items())):
                raise RuntimeError('Original clock or source integrity failed')
            parent_file=root/'small-core-mixing-tune-v2/report.json'
            if parent_file.exists():break
            write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_ORIGINAL_FINAL_AUDIT_AND_TERMINAL_RECEIPT',assigned_chips=0))
            time.sleep(5)
        parent=json.loads(parent_file.read_text());fits=completed_fits(source,parent)
        if parent['plan_sha256']!=plan['source_tune_plan_sha256']:
            raise AssertionError('Stopped parent identity differs')
        for key,row in fits.items():
            saved=next(r for r in row['checkpoint_records'] if r['update']==576)
            closure=root/'snapshots'/row['identity']/'closure.json'
            if (json.loads(closure.read_text())['exit_code']!=0
                    or file_sha256(root/'runs'/row['identity']/'checkpoint-u576.pt')!=saved['checkpoint_sha256']):
                raise AssertionError('Completed immutable native endpoint required')
        execution=Execution(root,file,out/'execution')
        execution.deadline=min(execution.deadline,plan['original_deadline_unix'])
        execution.execution['deadline_unix']=execution.deadline
        write_state(out/'execution/execution-start.json',execution.execution)
        def score(key,row):
            checkpoint=next(c for c in row['checkpoint_records'] if c['update']==576)
            target=out/'scores'/key
            checked=execution.cpu(key+'-score','scripts.evaluate_mixing_control_cpu',
                score_arguments(root,source_file,plan['source_tune_plan_sha256'],key,row['identity'],checkpoint['checkpoint_sha256'],target),
                target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
            if (checked['case_id']!=key or checked['training_update']!=576 or checked['depth']!=2
                    or checked['checkpoint_sha256']!=checkpoint['checkpoint_sha256']
                    or checked['training_real_targets']!=source['cases'][key]['end_targets']
                    or checked['estimated_training_dense_matrix_ops']!=source['cases'][key]['training_dense_matrix_ops']):
                raise AssertionError('Original cooled tuning state/source/work differs')
            return {**checked,'peak_lr':source['cases'][key]['peak_lr'],'model_variant':source['model_variant']}
        with cf.ThreadPoolExecutor(3) as pool:
            for future in [pool.submit(score,k,row) for k,row in fits.items()]:
                scores.append(future.result(timeout=max(1,execution.deadline-time.time())))
        if len(scores)!=3 or {s['case_id'] for s in scores}!=set(source['cases']):
            raise AssertionError('Every original trial must be scored')
        winner=min(scores,key=lambda s:(s['metrics']['weighted_nll'],s['peak_lr']))
        execution.check()
        result=dict(status='COMPLETE_CHEAP_MIXING_TUNE_SCORING_RECOVERY',scores=scores,
            gradient_gates=parent['gradient_gates'],selected_lr=winner['peak_lr'],
            selected_tune_nll=winner['metrics']['weighted_nll'],selected_at_grid_edge=winner['peak_lr'] in [.008,.032],
            source_tune_plan_sha256=plan['source_tune_plan_sha256'],stopped_parent_report_sha256=file_sha256(parent_file),
            original_fit_and_audit_costs_sha256=file_sha256(Path('results/small_core_mixing_tune_20261025/recovery-v2/execution-costs.json')))
    except BaseException as error:
        if execution:execution.failed.set()
        result.update(error=str(error),scores=scores);raise
    finally:
        if execution:execution.close(PUBLIC/'execution-costs.json')
        result.update(plan_sha256=file_sha256(file),optimizer_updates=0,new_training_matrix_ops=0,
            test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


if __name__=='__main__':main()
