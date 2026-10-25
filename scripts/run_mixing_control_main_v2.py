"""Same declared MAIN controller,with explicit complete score-recovery evidence."""
import argparse
import json
from pathlib import Path

from compute_trainium.training import file_sha256
from compute_trainium.future_span_execution import Execution
from scripts import run_mixing_control_main as engine
from scripts.register_mixing_control_main_v2 import register,IDENTITY,PUBLIC


class PredictionExecution(Execution):
    def cpu(self,identity,module,options,report_file,status):
        if module=='scripts.evaluate_mixing_control_cpu':module='scripts.evaluate_mixing_prediction_v2'
        return super().cpu(identity,module,options,report_file,status)


def run(root,plan,*,resume=False):
    source=root/'small-core-mixing-tune-scoring-v4/report.json'
    if (file_sha256(source)!=plan['selected_tune_recovery_report_sha256']
            or json.loads(source.read_text())['status']!='COMPLETE_CHEAP_MIXING_TUNE_SCORING_RECOVERY'):
        raise AssertionError('Complete unchanged recovery evidence required')
    engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.Execution=PredictionExecution
    engine.run(root,plan,resume=resume)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();file=args.root/IDENTITY/'plan.json'
    plan=json.loads(file.read_text()) if file.exists() else register(args.root)
    if not args.prepare_only:run(args.root,plan,resume=args.resume)


if __name__=='__main__':main()
