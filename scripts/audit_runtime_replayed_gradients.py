"""Join independently captured CPU vectors to fresh-process replay vectors."""
import argparse
import json
import time
from pathlib import Path

import torch

from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    started=time.time();root=a.root/'parcae-runtime-scalars-qualification-v1'
    public=Path('results/parcae_runtime_scalars_20261024');output=public/'replayed-gradient-report.json'
    if output.exists():raise FileExistsError('Preserve completed replay-vector audit')
    report_file=public/'report.json';report=json.loads(report_file.read_text())
    if report['status']!='PASS_COMPLETE_PARCAE_RUNTIME_SCALAR_QUALIFICATION':raise ValueError('Primary/recovery qualification incomplete')
    rows=[]
    for arm in ['ordinary','fixed1','poisson2']:
        row=next(r for r in report['records'] if r['arm']==arm)
        replay=a.root/'runs'/f'parcae-runtime-scalars-v1-{arm}-replay'
        native=json.loads((replay/'report.json').read_text())
        cpu_file=root/'cpu'/arm/'cpu-global.pt';native_file=replay/'updated-global-native.pt'
        if file_sha256(cpu_file)!=row['independent_trained_gradient']['cpu_global_gradient_sha256'] or file_sha256(native_file)!=native['updated_global_gradient_sha256']:
            raise AssertionError('Independent/replayed vectors changed')
        expected=torch.load(cpu_file,map_location='cpu',mmap=True,weights_only=False)
        actual=torch.load(native_file,map_location='cpu',mmap=True,weights_only=False)
        if any(expected[k]!=actual[k] for k in ['real_targets','source_cursor']):raise AssertionError('Replayed source objective differs')
        delta=gradient_comparison(expected['gradients'],actual['gradients']);difference=actual['loss']-expected['loss']
        rows.append({'arm':arm,'gradient':delta,'native_minus_cpu_nll':difference})
        if not delta['passes_two_percent_gate'] or abs(difference)>.003:raise AssertionError('Replayed raw global gradient misses independent CPU gate')
        del expected,actual
    result={'status':'PASS_RUNTIME_REPLAYED_INDEPENDENT_GRADIENTS','records':rows,
        'primary_qualification_sha256':file_sha256(report_file),
        'source_sha256':{f:file_sha256(f) for f in ['scripts/audit_runtime_replayed_gradients.py','compute_trainium/parcae_training_candidate.py']},
        'wall_seconds':time.time()-started,'assigned_chips':0,'optimizer_updates':0,'research_optimizer_updates':0,
        'scope':'All three fresh-process raw complete-global-objective gradients independently match pinned upstream CPU vectors at the exact recovered120-update point. No new native update or held-out scoring.'}
    write_state(output,result);print(json.dumps({'status':result['status'],'seconds':result['wall_seconds']}))


if __name__=='__main__':main()
