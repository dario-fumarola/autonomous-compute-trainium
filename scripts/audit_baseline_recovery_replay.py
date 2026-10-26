"""Check new replay identities against exact completed original warm states."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TORCH_XLA','0')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--warm',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--gold',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    import torch
    from compute_trainium.queue import write_state
    from compute_trainium.training import file_sha256
    from compute_trainium.parcae_training_candidate import gradient_comparison
    from scripts.run_parcae_resumed_training_qualification import check_native_pair
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Explicit numerical recovery plan changed')
    plan=json.loads(a.plan.read_text());profile=plan['profiles'][a.profile]
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Exact checker/model source changed')
    warm=json.loads((a.warm/'report.json').read_text());replay=json.loads((a.replay/'report.json').read_text())
    gold=json.loads((a.gold/'report.json').read_text())
    expected_origin=plan['inherited'][a.profile]['managed_plan_sha256'] if a.profile in plan['inherited'] else a.plan_sha
    if (warm['managed_qualification_plan_sha256']!=expected_origin or warm['qualification_profile']!=a.profile
            or replay['managed_qualification_plan_sha256']!=a.plan_sha or replay['qualification_profile']!=a.profile
            or gold['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT'
            or gold['checkpoint_sha256']!=warm['checkpoint_sha256']):raise ValueError('Exact old warm,new replay and independent point required')
    for path,sha in [(a.gold/'cpu-global.pt',gold['cpu_global_gradient_sha256']),
                     (a.replay/'updated-global-native.pt',replay['updated_global_gradient_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Independent/replay bytes changed')
    pair=check_native_pair(a.warm,a.replay)
    if not pair['passes_recovery_gate'] or pair['parameters']!=profile['parameters']:raise AssertionError('Original named update/moment recovery gate failed')
    expected=torch.load(a.gold/'cpu-global.pt',map_location='cpu',mmap=True,weights_only=False)
    actual=torch.load(a.replay/'updated-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
    if (expected['source_cursor'],expected['real_targets'])!=(actual['source_cursor'],actual['real_targets']):raise AssertionError('Replay source objective differs')
    delta=gradient_comparison(expected['gradients'],actual['gradients']);difference=actual['loss']-expected['loss']
    if not delta['passes_two_percent_gate'] or abs(difference)>.003:raise AssertionError('Original independent-gradient gate failed')
    a.output.mkdir(parents=True,exist_ok=False);write_state(a.output/'report.json',dict(
        status='PASS_BASELINE_RECOVERY_NAMED_REPLAY_AND_GLOBAL_GRADIENT',profile=a.profile,plan_sha256=a.plan_sha,
        original_warm_plan_sha256=expected_origin,pair=pair,gradient=delta,replay_minus_cpu_nll=difference,
        serialized_future_checkpoint_bytes_equal=replay['final_checkpoint_sha256']==warm['final_checkpoint_sha256'],
        assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
        scope='Exact completed warm state/signature,original2% named update/moment and independent512-window gradient criteria. New phase identities;future trajectory identity remains unestablished.'))


if __name__=='__main__':main()
