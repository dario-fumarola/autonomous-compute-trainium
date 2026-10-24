"""Validate retained complete native profiles after an independent-checker fix."""
import json

from .training import file_sha256


def retained_primary_proofs(root):
    retained={}
    identities=[('fp32-a16-ordinary','parcae-low-qual-v3-fp32-a16-ordinary-warm'),
                ('fp32-a8-poisson2','parcae-low-qual-v3-fp32-a8-poisson2-warm')]
    for key,identity in identities:
        path=root/'runs'/identity
        report=json.loads((path/'report.json').read_text())
        signature=report['signature']
        if (report['status']!='CAPTURED_PARCAE_TRAINING_QUALIFICATION' or report['mode']!='warm'
            or report['actual_optimizer_calls_per_replica']!=121 or report['warm_updates']!=100
            or not report['qualification_only'] or not report['all_replica_model_optimizer_states_equal']
            or signature['world']!=32 or signature['context']!=1024
            or f"{report['precision']}-a{report['accumulation']}-{report['arm']}"!=key):
            raise AssertionError('Only complete original native profiles may be retained')
        if any(file_sha256(f)!=sha for f,sha in signature['training_source_sha256'].items()):
            raise AssertionError('Retained native training arithmetic/source changed')
        artifacts={'report.json':file_sha256(path/'report.json')}
        files={'after-120-full-state.pt':'checkpoint_sha256',
               'after-121-model-optimizer.pt':'final_checkpoint_sha256',
               'initial-global-native.pt':'initial_global_gradient_sha256',
               'after-1-model-optimizer.pt':'first_checkpoint_sha256',
               'updated-global-native.pt':'updated_global_gradient_sha256','probe-depths.npy':'probe_depths_sha256'}
        if report['arm']!='ordinary':files['probe-initial-states.npy']='probe_initial_states_sha256'
        for name,field in files.items():
            sha=file_sha256(path/name)
            if sha!=report[field]:raise AssertionError('Original native profile artifact changed')
            artifacts[name]=sha
        retained[key]={'identity':identity,'artifact_sha256':artifacts,
                       'retained_optimizer_calls_per_replica':121,'warm_updates':100,
                       'original_costs':'results/parcae_low_depth_qualification_20261024/attempt3/execution-costs.json'}
    return retained
