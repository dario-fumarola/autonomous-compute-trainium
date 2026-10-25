"""CPU-only qualification of immutable stable-state plan-identity transport."""
import copy
import json
import shutil
import time
import argparse
from pathlib import Path

import numpy as np
import torch

from compute_trainium.future_span_transport import transport_stable
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint, file_sha256
from scripts.qualify_native import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    identity = 'future-span-transport-qualification-v1'
    out = args.root / identity
    public = Path('results/future_span_transport_20261025')
    if out.exists() or public.exists():
        raise FileExistsError('Preserve an existing transport qualification')
    out.mkdir()
    public.mkdir(parents=True)
    result = {'status': 'STOPPED_CPU_FUTURE_SPAN_STATE_TRANSPORT', 'assigned_chips': 0,
              'model_forwards': 0, 'optimizer_updates': 0, 'test_accessed': False}
    try:
        source_file = args.root / 'future-span-main-v2/plan.json'
        source_plan = json.loads(source_file.read_text())
        source_sha = file_sha256(source_file)
        if source_plan['identity'] != 'future-span-main-v2':
            raise ValueError('Exact existing MAIN registration required')
        paths = ['scripts/qualify_future_span_transport.py',
                 'compute_trainium/future_span_transport.py',
                 'tests/test_future_span_transport.py']
        sources = dict(source_plan['source_sha256'])
        for name, sha in sources.items():
            if file_sha256(name) != sha:
                raise AssertionError('Original frozen scientific source changed')
        sources.update({p: file_sha256(p) for p in paths})
        readiness_file = args.root / 'future-span-compute-readiness-v1/report.json'
        readiness = json.loads(readiness_file.read_text())
        if (readiness['status'] != 'PASS_PROSPECTIVE_WORK_MATCHED_CONTINUATION_GEOMETRY'
                or readiness['main_plan_sha256'] != source_sha):
            raise ValueError('Same-plan work/source geometry audit required')
        for name, sha in readiness['analysis_source_sha256'].items():
            if file_sha256(name) != sha:
                raise AssertionError('Prospective geometry analysis source changed')
            sources[name] = sha
        geometry = readiness['one_pass_control_branch']
        origin = geometry['origin_update']
        source_keys = [key for key, case in source_plan['cases'].items()
                       if case['seed'] == 5231 and case['arm'] == 'one']
        if len(source_keys) != 2:
            raise ValueError('Both trained objective cells required')
        original_files = {key: args.root / 'runs' /
            ('future-span-main-v2-' + key + '-first') / f'checkpoint-u{origin}.pt'
            for key in source_keys}
        reserve = 35*1024**3 + 2*sum(p.stat().st_size for p in original_files.values())
        available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:')))
        if shutil.disk_usage(out).free < reserve or available_kib < 512*1024**2:
            raise RuntimeError('Keep the established memory and disk reserves')
        parents = {key: {'file': str(p.relative_to(args.root)), 'sha256': file_sha256(p)}
                   for key, p in original_files.items()}
        native_receipts = {}
        for key, file in original_files.items():
            report_file = file.parent / 'report.json'
            closure_file = args.root / 'snapshots' / file.parent.name / 'closure.json'
            report = json.loads(report_file.read_text())
            checkpoint = next(r for r in report['checkpoint_records'] if r['update'] == origin)
            if (report['status'] != 'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT'
                    or report['plan_sha256'] != source_sha or report['case_id'] != key
                    or checkpoint['checkpoint_sha256'] != parents[key]['sha256']
                    or not checkpoint['all_complete_replica_states_equal']
                    or json.loads(closure_file.read_text())['exit_code'] != 0):
                raise AssertionError('Original native checkpoint/closure provenance differs')
            native_receipts[key] = {'report_sha256': file_sha256(report_file),
                'closure_sha256': file_sha256(closure_file),
                'complete_model_optimizer_digest': checkpoint['full_state_digest']}
        packing = args.root / source_plan['data']['packed']
        packing_manifest = json.loads((packing / 'manifest.json').read_text())
        count_file = packing / 'prefix_counts.npy'
        if (file_sha256(packing / 'manifest.json') != source_plan['data']['manifest_sha256']
                or file_sha256(count_file) != packing_manifest['files']['prefix_counts.npy']):
            raise AssertionError('Original natural source counter file changed')
        counts = np.load(count_file, mmap_mode='r')
        original_domains = list(map(int, counts[origin*512] - counts[0]))
        if original_domains != geometry['origin_source']['domain_targets']:
            raise AssertionError('Prospective geometry and original source counters differ')
        aux = source_plan['auxiliary_data']
        anchor_file = args.root / aux['directory'] / aux['prefix_file']
        if file_sha256(anchor_file) != aux['prefix_sha256']:
            raise AssertionError('Original auxiliary prefix changed')
        anchors = np.load(anchor_file, mmap_mode='r')
        destination = copy.deepcopy(source_plan)
        destination.update(identity=identity, stage='cpu_state_transport_qualification',
            registered_unix=time.time(), launch_gates_complete=False, root_comm_ports={},
            source_sha256=sources, cases={}, transport_parent_checkpoints=parents,
            source_plan_sha256=source_sha, execution_allowed=False,
            scope='CPU metadata/tensor-copy qualification only. Native launch is disabled. Not a registered scientific continuation or evidence of restart trajectory agreement.')
        destination['worker_source_sha256'] = {p: sha for p, sha in sources.items()
            if p.startswith(('compute_trainium/', 'scripts/', 'configs/'))}
        for key in source_keys:
            child = copy.deepcopy(source_plan['cases'][key])
            child.update(parent={'case_id': key, 'update': origin},
                total_updates=geometry['total_updates'], end_targets=geometry['final_source']['real_targets'],
                end_anchors=int(anchors[geometry['final_source']['source_cursor']]),
                cooldown_start_update=geometry['cooldown_start_update'],
                cooldown_start_targets=geometry['cooldown_source']['real_targets'],
                training_dense_matrix_ops=geometry['matched_work']['executed_matmuls'],
                checkpoint_updates=[], segment_ends=[], gradient_updates=[])
            destination['cases'][key + '-transport-qualification'] = child
        destination_file = out / 'nonlaunchable-plan.json'
        write_state(destination_file, destination)
        destination_sha = file_sha256(destination_file)
        rows = []
        for key, file in original_files.items():
            saved = torch.load(file, map_location='cpu', mmap=True, weights_only=False)
            case = source_plan['cases'][key]
            work = np.load(source_file.parent / case['work_table'], mmap_mode='r')
            if file_sha256(source_file.parent / case['work_table']) != case['work_table_sha256']:
                raise AssertionError('Original work ledger changed')
            counters = dict(expected_targets=geometry['origin_source']['real_targets'],
                expected_domains=geometry['origin_source']['domain_targets'],
                expected_matrix_ops=sum(map(int, work[:origin])),
                expected_anchors=int(anchors[geometry['origin_source']['source_cursor']]))
            new, proof = transport_stable(saved, source_plan=source_plan, source_plan_sha=source_sha,
                source_case_id=key, source_checkpoint_sha=parents[key]['sha256'],
                destination_plan=destination, destination_plan_sha=destination_sha,
                destination_case_id=key + '-transport-qualification', expected_counters=counters)
            if proof['complete_model_optimizer_digest'] != native_receipts[key]['complete_model_optimizer_digest']:
                raise AssertionError('Loaded complete state differs from the original native receipt')
            target = out / (key + '.pt')
            atomic_checkpoint(target, new)
            restored = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
            if (digest({'model': dict(sorted(restored['model'].items())), 'optimizer': restored['optimizer']})
                    != proof['complete_model_optimizer_digest']
                    or digest(restored['cpu_rng_by_rank']) != proof['cpu_rng_digest']
                    or restored['signature'] != new['signature']
                    or restored['continuation_provenance'] != new['continuation_provenance']
                    or file_sha256(file) != parents[key]['sha256']):
                raise AssertionError('Serialized copy or immutable original differs')
            rows.append({'case_id': key, **proof, 'derived_checkpoint_sha256': file_sha256(target),
                'serialized_model_optimizer_rng_exact': True, 'original_file_unchanged': True,
                'source_schedule_retained': restored['schedule'] == saved['schedule'],
                'original_native_receipts': native_receipts[key]})
            del saved, new, restored
            if time.time() - started > 180:
                raise TimeoutError('Fixed180-second CPU transport qualification cap')
        result.update(status='PASS_CPU_FUTURE_SPAN_STATE_TRANSPORT', records=rows,
            source_plan_sha256=source_sha, destination_plan_sha256=destination_sha,
            geometry_receipt_sha256=file_sha256(readiness_file),
            analysis_source_sha256={p: sources[p] for p in paths},
            original_frozen_source_count=len(source_plan['source_sha256']),
            native_launch_enabled=False, scientific_continuation_admitted=False,
            scope=destination['scope'])
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        result['wall_seconds'] = time.time() - started
        write_state(out / 'report.json', result)
        write_state(public / 'report.json', result)
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
