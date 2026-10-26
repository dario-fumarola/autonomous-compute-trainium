"""Metadata preparation for one conventional numerical control;no launch."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_control_replication import reserves

IDENTITY='ordinary8-native-preparation-v1'
PUBLIC=Path('results/ordinary8_native_20261026/preparation')
DOC='docs/ORDINARY8_NATIVE_PREPARATION_20261026.md'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/IDENTITY
    if out.exists():raise FileExistsError('Preserve numerical preparation')
    reserves(a.root)
    parity_dir=a.root/'ordinary8-cpu-parity-v2';parity=json.loads((parity_dir/'report.json').read_text())
    if (parity['status']!='PASS_ORDINARY8_CPU_MODEL_PARITY' or parity['parameters']!=141733120
            or parity['upstream_n_layer']!=8 or parity['target_optimizer_steps']!=0):raise ValueError('Exact completed eight-layer CPU parity required')
    initial=parity_dir/'initial-state-ordinary.pt'
    if file_sha256(initial)!=parity['initializer_sha256']:raise AssertionError('Qualification C0 initializer changed')
    old_dir=a.root/'lean-ntp-native-preparation-v1';old=json.loads((old_dir/'plan.json').read_text())
    sources={**old['source_sha256'],**parity['source_sha256']}
    for file in ['scripts/prepare_ordinary8_native.py','scripts/qualify_ordinary8_managed.py',
                 'scripts/audit_ordinary8_native.py','scripts/run_parcae_resumed_training_qualification.py',
                 'tests/test_ordinary8_native.py',DOC]:sources[file]=file_sha256(file)
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Independent/qualified source freeze changed')
    worker,host=split_snapshot_sources(sources);plan=copy.deepcopy(old)
    out.mkdir();(out/'initial-state-ordinary.pt').symlink_to(os.path.relpath(initial,out))
    values=np.ones((121,16,32,2),dtype=np.uint16);depth=out/'ordinary-a16-depths.npy';np.save(depth,values)
    if not np.array_equal(np.load(depth),values):raise AssertionError('True one-pass table readback differs')
    plan.update(identity=IDENTITY,registered_unix=started,model=parity['model'],
        execution_variant='ordinary8_fp16_primitive_attention_runtime_scalars',
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        initial_states=dict(ordinary=dict(file='initial-state-ordinary.pt',sha256=parity['initializer_sha256'],
            parameters=141733120,model_digest=parity['initial_model_digest'])),
        cases=[dict(arm='ordinary',accumulation=16,depth_table_sha256=file_sha256(depth),initial_kind='ordinary',
                    minimum_depth=1,maximum_depth=1,mean_depth=1.,mean_backward_tail=1.)],
        independent_cpu_parity_sha256=file_sha256(parity_dir/'report.json'),
        original_numerical_preparation_sha256=file_sha256(old_dir/'plan.json'),
        qualification_only=True,native_launch_allowed=False,scientific_main_admitted=False,scope=Path(DOC).read_text())
    write_state(out/'plan.json',plan)
    write_state(out/'cpu-report.json',dict(status='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION',
        scope='Compatibility marker for the exact eight-layer pinned-upstream CPU parity. No model forward,optimizer update or native qualification in this preparation.'))
    view=out/'audit-root';view.mkdir();(view/'parcae-low-depth-qualification-v1').symlink_to('..')
    (view/'natural-append-6500m-v2').symlink_to('../../natural-append-6500m-v2')
    if ((view/'parcae-low-depth-qualification-v1/plan.json').resolve()!=(out/'plan.json').resolve()
            or (view/'natural-append-6500m-v2').resolve()!=(a.root/'natural-append-6500m-v2').resolve()):raise AssertionError('Explicit independent auditor view differs')
    report=dict(status='PASS_ORDINARY8_NATIVE_INPUT_PREPARATION_ONLY',plan_sha256=file_sha256(out/'plan.json'),
        parameters=141733120,model_forwards=0,optimizer_updates=0,assigned_chips=0,
        qualification_only=True,native_launch_allowed=False,scientific_fit_admitted=False,test_accessed=False,
        terminal_alpha_spent=0,wall_seconds=time.time()-started,
        audit_view=dict(preparation='This exact prepared ordinary8 plan and unchanged qualification initializer',
                        data='Original natural-append-6500m-v2 training source only'),scope=Path(DOC).read_text())
    write_state(out/'report.json',report);PUBLIC.mkdir(parents=True);write_state(PUBLIC/'plan.json',plan);write_state(PUBLIC/'report.json',report)


if __name__=='__main__':main()
