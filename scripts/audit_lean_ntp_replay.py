"""Independent-vector and named-state check of a qualification-only replay."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TORCH_XLA','0')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--warm',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--gold',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    from compute_trainium.parcae_training_candidate import gradient_comparison
    from compute_trainium.queue import write_state
    from compute_trainium.training import file_sha256
    from scripts.run_parcae_resumed_training_qualification import check_native_pair
    started=time.time();torch.set_num_threads(12);torch.set_num_interop_threads(1)
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Qualification plan changed')
    plan=json.loads(a.plan.read_text())
    if any(file_sha256(file)!=sha for file,sha in plan['source_sha256'].items()):raise AssertionError('Frozen checker changed')
    gold=json.loads((a.gold/'report.json').read_text())
    if gold['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':raise ValueError('Independent full-global gold required')
    native=json.loads((a.replay/'report.json').read_text())
    if native['managed_qualification_plan_sha256']!=a.plan_sha:raise AssertionError('Replay belongs to another qualification')
    for path,sha in [(a.gold/'cpu-global.pt',gold['cpu_global_gradient_sha256']),
                     (a.replay/'updated-global-native.pt',native['updated_global_gradient_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Independent/replayed gradient bytes changed')
    pair=check_native_pair(a.warm,a.replay)
    if not pair['passes_recovery_gate'] or pair['parameters']!=141963840:
        raise AssertionError('Complete named-state replay gate failed')
    expected=torch.load(a.gold/'cpu-global.pt',map_location='cpu',mmap=True,weights_only=False)
    actual=torch.load(a.replay/'updated-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
    if (expected['source_cursor'],expected['real_targets'])!=(actual['source_cursor'],actual['real_targets']):
        raise AssertionError('Exact global source denominator differs after replay')
    delta=gradient_comparison(expected['gradients'],actual['gradients'])
    difference=actual['loss']-expected['loss']
    if not delta['passes_two_percent_gate'] or abs(difference)>.003:
        raise AssertionError('Replayed complete gradient disagrees with independent FP32')
    a.output.mkdir(parents=True,exist_ok=False)
    write_state(a.output/'report.json',dict(status='PASS_LEAN_NTP_NAMED_REPLAY_AND_GLOBAL_GRADIENT',
        plan_sha256=a.plan_sha,pair=pair,replay_gradient=delta,replay_minus_cpu_nll=difference,
        serialized_future_checkpoint_bytes_equal=native['final_checkpoint_sha256']==json.loads((a.warm/'report.json').read_text())['final_checkpoint_sha256'],
        wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,test_accessed=False,
        scope='Exact loaded state/RNG and registered2% named update/moment replay gate;replayed512-window gradient against the independent upstream FP32 vector. Bit equality is reported separately and does not establish identical future trajectories.'))


if __name__=='__main__':main()
