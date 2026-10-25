"""Explicitly scope reuse of the fixed read-only answer evaluator on TRAIN."""
import json
from pathlib import Path

def verify_train_panel(arguments):
    path=Path(arguments[arguments.index('--plan')+1]);plan=json.loads(path.read_text())
    if plan.get('panel_role')!='train_only_post_pilot_diagnostic':raise ValueError('Explicit TRAIN-only diagnostic plan required')


def mark_train_report(arguments):
    output=Path(arguments[arguments.index('--output')+1])/'report.json';report=json.loads(output.read_text())
    report.update(panel_role='train_only_post_pilot_diagnostic',heldout_result=False,
        scope='Post-pilot TRAIN-only frozen-checkpoint diagnosis with original no-work source views and unchanged prefix-only greedy/scalar readout. No model update,held-out replacement,efficacy or terminal claim. '+('Independent FP32 answer-loss check.' if 'INDEPENDENT' in report['status'] else 'Native prediction weights/replicas unchanged.'))
    output.write_text(json.dumps(report,indent=2)+'\n')
