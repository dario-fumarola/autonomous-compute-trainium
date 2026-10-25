"""Zero-chip fixed queue:complete fair TUNE gates,then admit declared MAIN."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='small-core-mixing-main-queue-v1'
PUBLIC=Path('results/small_core_mixing_main_queue_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    out=args.root/IDENTITY;file=out/'plan.json'
    tune_file=args.root/'small-core-mixing-tune-v2/plan.json'
    if not file.exists():
        tune=json.loads(tune_file.read_text())
        additions=['scripts/register_mixing_control_main.py','scripts/run_mixing_control_main.py',
                   'scripts/queue_mixing_control_main.py']
        plan=dict(identity=IDENTITY,registered_unix=time.time(),tune_plan_sha256=file_sha256(tune_file),
            source_sha256={**tune['source_sha256'],**{p:file_sha256(p) for p in additions}},
            prerequisite_deadline_unix=tune['original_absolute_deadline_unix']+1800,
            expected_tune_status='COMPLETE_BOUNDED_CHEAP_MIXING_TUNE',assigned_chips=0,
            scope='Fixed zero-chip queue for the already declared two-seed control endpoints. All three complete TUNE opportunities and six numerical gates required;no favorable-loss threshold or new variant selection. MAIN separately validates selected rate,untouched original random states,source/work geometry,complete measured3-hour budget and idle ownership. No main fit if TUNE stops,hashes change or original predecessor window expires. No future exposure extension,TEST or terminal alpha.')
        out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
        write_state(file,plan);write_state(PUBLIC/'plan.json',plan)
    plan=json.loads(file.read_text())
    if args.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal queue')
    result=dict(status='STOPPED_CHEAP_CONTROL_MAIN_QUEUE',assigned_chips=0,test_accessed=False)
    try:
        while True:
            if (file_sha256(tune_file)!=plan['tune_plan_sha256']
                    or any(file_sha256(p)!=h for p,h in plan['source_sha256'].items())):
                raise AssertionError('Frozen control queue source/prerequisite changed')
            if time.time()>plan['prerequisite_deadline_unix']:
                raise TimeoutError('Original finite predecessor window expired')
            target=args.root/'small-core-mixing-tune-v2/report.json'
            if target.exists():
                prior=json.loads(target.read_text())
                if prior['status']!=plan['expected_tune_status']:
                    raise RuntimeError('TUNE did not complete;no automatic MAIN')
                if not managed_running():break
            write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_COMPLETE_THREE_RATE_TUNE_GATES',assigned_chips=0))
            time.sleep(10)
        from scripts.register_mixing_control_main import register
        from scripts.run_mixing_control_main import run
        next_plan=register(args.root)
        write_state(PUBLIC/'status.json',dict(status='MAIN_ADMITTED_WITH_OWN_MEASURED_BUDGET',maximum_chips=16))
        run(args.root,next_plan)
        next_file=args.root/'small-core-mixing-main-v1/report.json'
        result=dict(status='COMPLETE_DECLARED_CHEAP_CONTROL_CHAIN',main_report_sha256=file_sha256(next_file))
    except BaseException as error:
        result.update(error=str(error));raise
    finally:
        result.update(plan_sha256=file_sha256(file),test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':main()
