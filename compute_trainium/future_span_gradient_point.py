"""Exact identities/counters for zero-update scientific-checkpoint probes."""
import json
from pathlib import Path

import numpy as np

from .future_span_scientific_state import SCIENCE,signature
from .parcae_scientific_state import schedule_identity
from .training import file_sha256


def validate_point(saved,plan_sha,plan,case_id,domains,anchors,work):
    case=plan['cases'][case_id];update=saved['update']
    if (plan['purpose']!=SCIENCE or saved['purpose']!=SCIENCE or saved['qualification_only'] is not False
            or saved['case_id']!=case_id or saved['signature']!=signature(plan_sha,plan,case)
            or saved['schedule']!=schedule_identity(case)):
        raise ValueError('Exact scientific objective/seed/source/schedule identity required')
    if update not in case['gradient_updates'] or not 0<update<=case['total_updates']:
        raise ValueError('Unregistered mature-gradient checkpoint')
    if (saved['cursor']!=update*512 or saved['domain_targets']!=list(map(int,domains))
            or saved['real_targets']!=sum(map(int,domains)) or saved['valid_anchors']!=int(anchors)
            or saved['estimated_dense_matrix_ops']!=int(work)
            or saved['future_source_position_presentations']!=int(anchors)*31
            or saved['executed_head_slots']!=update*512*63
            or saved['stable_trunk']!=(saved['real_targets']<=case['cooldown_start_targets'])):
        raise ValueError('Exact mature source/domain/auxiliary/work counters required')
    return case


def point_tables(root,plan_file,plan,case):
    root,plan_file=Path(root),Path(plan_file);packing=root/plan['data']['packed']
    meta_file=packing/'manifest.json';meta=json.loads(meta_file.read_text())
    if file_sha256(meta_file)!=plan['data']['manifest_sha256']:raise AssertionError('Natural packing changed')
    count_file=packing/'prefix_counts.npy'
    if file_sha256(count_file)!=meta['files']['prefix_counts.npy']:raise AssertionError('Natural source counts changed')
    aux=plan['auxiliary_data'];anchor_file=root/aux['directory']/aux['prefix_file']
    depth_file=plan_file.parent/case['depth_table'];work_file=plan_file.parent/case['work_table']
    for path,sha in [(anchor_file,aux['prefix_sha256']),(depth_file,case['depth_table_sha256']),
                     (work_file,case['work_table_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Registered mature-probe tables changed')
    return tuple(np.load(p,mmap_mode='r') for p in [count_file,anchor_file,depth_file,work_file])
