"""Metadata-only preparation; no native admission or scientific initializer."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='lean-ntp-native-preparation-v1'
PUBLIC=Path('results/lean_ntp_native_20261026/preparation')
DOC='docs/LEAN_NTP_NATIVE_PREPARATION_20261026.md'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/IDENTITY
    if out.exists():raise FileExistsError('Preserve prepared numerical inputs')
    require_inode_reserve([a.root,Path('/')])
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<544*1024**3 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Protected CPU preparation reserves failed')
    old_dir=a.root/'parcae-low-depth-qualification-v1';old_file=old_dir/'plan.json';old=json.loads(old_file.read_text())
    old_cpu=old_dir/'cpu-report.json';parity=a.root/'lean-ntp-cpu-parity-v1/cpu/report.json'
    public_file=Path('results/lean_ntp_cpu_parity_20261025/report.json')
    public_parity=json.loads(public_file.read_text())
    if (public_parity['status']!='PASS_LEAN_NTP_CPU_PARITY'
            or file_sha256(parity)!=file_sha256(public_file)
            or file_sha256('scripts/check_lean_ntp_cpu.py')!=public_parity['source_sha256']):
        raise ValueError('Unchanged completed CPU head-removal parity required')
    if json.loads(old_cpu.read_text())['status']!='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION':
        raise ValueError('Original complete lean qualification preparation required')
    parent_file=a.root/'fixed-control-replication-main-v1/plan.json';parent=json.loads(parent_file.read_text())
    full_child=json.loads((parent_file.parent/parent['child_plans']['full']['file']).read_text())
    if old['model']!=full_child['model'] or old['source_corpus_sha256']!=full_child['data']['corpus_sha256']:
        raise AssertionError('Original lean model/source must match the active family')
    sources=dict(parent['source_sha256'])
    for file in ['scripts/prepare_lean_ntp_native.py','scripts/qualify_lean_ntp_primitives.py',DOC,
                 'scripts/audit_parcae_precise_training_point.py','compute_trainium/parcae_recovery_check.py',
                 'scripts/check_lean_ntp_cpu.py','compute_trainium/lean_ntp_admission.py','tests/test_lean_ntp_admission.py']:
        sha=file_sha256(file)
        if file in sources and sources[file]!=sha:raise AssertionError('Frozen numerical source conflicts')
        sources[file]=sha
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Qualified numerical source changed')
    initial=copy.deepcopy(old['initial_states']['family']);initial_file=(old_dir/initial['file']).resolve()
    if file_sha256(initial_file)!=initial['sha256'] or initial['parameters']!=141963840:
        raise AssertionError('Original qualification-only random initializer changed')
    initial['file']=os.path.relpath(initial_file,out)
    corpus=a.root/full_child['data']['corpus'];manifest=a.root/full_child['data']['packed']/'manifest.json'
    if (file_sha256(corpus)!=old['source_corpus_sha256']
            or file_sha256(manifest)!=old['packed_manifest_sha256']
            or old['packing_fingerprint']!=full_child['data']['packing_fingerprint']):
        raise AssertionError('Original authentic source/packing identity changed')
    worker,host=split_snapshot_sources(sources);plan=copy.deepcopy(old)
    plan.update(identity=IDENTITY,registered_unix=started,
        execution_variant='lean_fp16_primitive_attention_runtime_scalars',
        qualification_schedule='existing_changing_qualification_schedule_peak_1e-4',
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        initial_states={'family':initial},cases=[],qualification_only=True,
        reference_preparation_plan_sha256=file_sha256(old_file),
        reference_cpu_report_sha256=file_sha256(old_cpu),cpu_head_removal_parity=public_parity,
        cpu_head_removal_private_report_sha256=file_sha256(parity),
        current_main_plan_sha256=file_sha256(parent_file),scientific_main_admitted=False,
        native_launch_allowed=False,scope=Path(DOC).read_text())
    out.mkdir();records=[]
    for legacy,depth in [('fixed1',1),('poisson2',2)]:
        values=np.full((121,16,32,2),depth,dtype=np.uint16);file=out/f'{legacy}-a16-depths.npy';np.save(file,values)
        if not np.array_equal(np.load(file,mmap_mode='r'),values):raise AssertionError('Forced-depth readback failed')
        plan['cases'].append(dict(arm=legacy,accumulation=16,depth_table_sha256=file_sha256(file),
                                 initial_kind='family',minimum_depth=depth,maximum_depth=depth,
                                 mean_depth=float(depth),mean_backward_tail=float(depth)))
        parity_row=next(v for v in public_parity['records'] if v['depth']==depth)
        records.append(dict(label='one' if depth==1 else 'fixed2',legacy_worker_selector=legacy,
                            forward_depth=depth,backward_depth=depth,
                            inherited_cpu_work_ledger=parity_row['lean_global_update_work']))
    if time.time()-started>180:raise TimeoutError('CPU preparation cap')
    write_state(out/'plan.json',plan)
    write_state(out/'cpu-report.json',dict(status='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION',
        inherited_original_cpu_report_sha256=file_sha256(old_cpu),
        scope='Compatibility marker:original independent CPU preparation plus completed head-removal parity. This metadata preparation performs no model forward or optimizer update.'))
    report=dict(status='PASS_LEAN_NTP_NATIVE_PREPARATION_ONLY',plan_sha256=file_sha256(out/'plan.json'),
                records=records,model_forwards=0,optimizer_updates=0,assigned_chips=0,
                native_launch_allowed=False,scientific_main_admitted=False,test_accessed=False,
                terminal_alpha_spent=0,wall_seconds=time.time()-started,
                scope='Fixed numerical inputs only;no new hardware qualification,scientific state or held-out result.')
    write_state(out/'report.json',report);PUBLIC.mkdir(parents=True)
    write_state(PUBLIC/'plan.json',plan);write_state(PUBLIC/'report.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
