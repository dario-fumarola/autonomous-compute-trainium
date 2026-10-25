"""Read-only hindsight bound after complete original work controls."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from compute_trainium.document_depth_oracle import weighted_document_losses, hindsight_point
from compute_trainium.parcae_candidate import ParcaeCandidateConfig, matmul_ledger
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY = 'future-span-depth-oracle-v2'
PUBLIC = Path('results/future_span_depth_oracle_20261025')
DOC = Path('docs/FULL_GRADIENT_DEPTH_ORACLE_20261025.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.root / IDENTITY
    plan_file = out / 'plan.json'
    main_file = args.root / 'future-span-main-v2/report.json'
    work_plan_file = args.root / 'future-span-work-control-v1/plan.json'
    work_dir = args.root / 'future-span-work-control-v1'
    main_plan_file = args.root / 'future-span-main-v2/plan.json'
    if not plan_file.exists():
        if (work_dir / 'scores/one-ntp-s7283-lr32000-work/report.json').exists():
            raise RuntimeError('Preserve disclosure: register before second pair scores')
        source_paths = [__file__, str(DOC), 'compute_trainium/document_depth_oracle.py',
                        'compute_trainium/parcae_candidate.py', 'tests/test_document_depth_oracle.py']
        plan = dict(identity=IDENTITY, registered_unix=time.time(),
            selected_after_first_seed_matched_work_negative=True,
            main_report_sha256=file_sha256(main_file),
            main_plan_sha256=file_sha256(main_plan_file),
            work_plan_sha256=file_sha256(work_plan_file),
            source_sha256={str(Path(p).relative_to(Path.cwd()) if Path(p).is_absolute() else Path(p)):file_sha256(p)
                           for p in source_paths},
            seeds=[5231,7283], depths=[1,2,3],
            weights={'web':.6,'math':.2,'code':.2},
            penalties=[0.]+list(map(float,np.geomspace(1e-4,10.,41))),
            seconds_cap=180, scope=DOC.read_text())
        out.mkdir()
        PUBLIC.mkdir(parents=True,exist_ok=True)
        write_state(plan_file,plan)
        write_state(PUBLIC / 'registration.json',plan)
    plan = json.loads(plan_file.read_text())
    if (plan['identity'] != IDENTITY or file_sha256(main_file) != plan['main_report_sha256']
            or file_sha256(main_plan_file) != plan['main_plan_sha256']
            or file_sha256(work_plan_file) != plan['work_plan_sha256']
            or any(file_sha256(p) != h for p,h in plan['source_sha256'].items())):
        raise AssertionError('Registered source or parent identity changed')
    if args.prepare_only:
        print(json.dumps({'status':'REGISTERED_ZERO_CHIP_RETROSPECTIVE_BOUND','plan_sha256':file_sha256(plan_file)}))
        return
    if (out / 'report.json').exists():
        raise FileExistsError('Preserve completed analysis')
    begin = time.time()
    main_report = json.loads(main_file.read_text())
    work_file = work_dir / 'report.json'
    work_report = json.loads(work_file.read_text())
    main_plan = json.loads(main_plan_file.read_text())
    work_plan = json.loads(work_plan_file.read_text())
    if (main_report['status'] != 'COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT'
            or work_report['status'] != 'COMPLETE_WITHIN_FAMILY_WORK_MATCHED_CONTROLS'
            or work_report['plan_sha256'] != plan['work_plan_sha256']
            or len(work_report['scores']) != 4 or len(work_report['gradient_gates']) != 8):
        raise AssertionError('Complete original studies and every numerical gate required')
    config = ParcaeCandidateConfig(**main_plan['model'])
    forward = {d:matmul_ledger(config,1024,recurrent=True,total_steps=d)['forward_matmul_flops']
               for d in plan['depths']}
    inputs, rows = [], []

    def load(folder, seed, depth, expected_case, expected_targets, expected_work):
        if time.time()-begin > plan['seconds_cap']:
            raise TimeoutError('Bounded CPU analysis expired')
        file, docs_file = folder / 'report.json', folder / 'document-scores.json'
        report = json.loads(file.read_text())
        if (report['status'] != 'COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE'
                or report['seed'] != seed or report['depth'] != depth
                or report['case_id'] != expected_case or report['cohort'] != 'curve'
                or report['split'] != 'dev' or report['training_real_targets'] != expected_targets
                or report['estimated_training_dense_matrix_ops'] != expected_work
                or report['deployed_parameters'] != config.parameter_count(True)
                or not all(report[k] for k in ['head_removed','model_unchanged','full_state_readback_exact'])
                or report['source_corpus_sha256'] != main_plan['evaluation']['corpus_sha256']):
            raise AssertionError('Original trained endpoint, panel or immutable state differs')
        docs = sorted(json.loads(docs_file.read_text())['records'],key=lambda r:r['id'])
        pairing = [(d['id'],d['group'],d['domain'],d['real_targets']) for d in docs]
        if len(set(d['id'] for d in docs)) != len(docs) or len(docs) != 603 or sum(d['real_targets'] for d in docs) != 754750:
            raise AssertionError('Original all-token document coverage differs')
        converted = [dict(domain=d['domain'],tokens=d['real_targets'],nll_sum=d['nll_sum']) for d in docs]
        loss = weighted_document_losses(converted,plan['weights'])
        if abs(float(loss.sum())-report['metrics']['weighted_nll']) > 1e-10:
            raise AssertionError('Independent weighted loss reaggregation differs')
        inputs.append(dict(identity=folder.name,report_sha256=file_sha256(file),
            document_scores_sha256=file_sha256(docs_file),checkpoint_sha256=report['checkpoint_sha256']))
        return report, docs, pairing, loss

    for seed in plan['seeds']:
        case_id = f'recurrent-ntp-s{seed}-lr32000'
        case = main_plan['cases'][case_id]
        losses, costs, pairing = [], [], None
        for depth in plan['depths']:
            report, docs, own_pairing, loss = load(args.root / 'future-span-main-v2/scores' / f'{case_id}-cooled1b-d{depth}',
                seed,depth,case_id,case['end_targets'],case['training_dense_matrix_ops'])
            if pairing is not None and own_pairing != pairing:
                raise AssertionError('Depth changed document/source-group/target pairing')
            pairing = own_pairing
            windows = np.array([math.ceil(d['real_targets']/1024) for d in docs])
            if int(windows.sum()) != report['windows']:
                raise AssertionError('Whole-document window accounting differs')
            losses.append(loss)
            costs.append(windows*forward[depth]/(windows.sum()*forward[1]))
        points = []
        for penalty in plan['penalties']:
            point = hindsight_point(np.stack(losses),np.stack(costs),penalty)
            choices = point.pop('choice')
            point.update(penalty=penalty,document_counts_by_depth={str(d):int((choices == i).sum())
                         for i,d in enumerate(plan['depths'])})
            points.append(point)
        controls = []
        for cell in ['one-ntp','one-future']:
            key = f'{cell}-s{seed}-lr32000-work'
            c = work_plan['cases'][key]
            r, _, control_pairing, loss = load(work_dir / 'scores' / key,seed,1,key,c['end_targets'],c['training_dense_matrix_ops'])
            if control_pairing != pairing or abs(c['training_dense_matrix_ops']/case['training_dense_matrix_ops']-1) > .001:
                raise AssertionError('Matched complete training work or document pairing differs')
            if not any(g['case_id'] == key and g['update'] == r['training_update'] for g in work_report['gradient_gates']):
                raise AssertionError('Missing complete original endpoint numerical audit')
            controls.append(dict(cell=cell,nll=float(loss.sum()),
                free_oracle_gain_over_control=float(loss.sum())-points[0]['weighted_nll']))
        rows.append(dict(seed=seed,constant_depth_nll={str(d):float(losses[i].sum()) for i,d in enumerate(plan['depths'])},
            points=points,controls=controls,documents=len(pairing),real_targets=sum(p[3] for p in pairing)))
    result = dict(status='COMPLETE_RETROSPECTIVE_FULL_GRADIENT_DOCUMENT_BOUND',records=rows,
        plan_sha256=file_sha256(plan_file),parent_work_report_sha256=file_sha256(work_file),input_hashes=inputs,
        fixed_window_forward_matrix_ops=forward,wall_seconds=time.time()-begin,new_model_forwards=0,
        optimizer_updates=0,assigned_chips=0,test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
    write_state(out / 'report.json',result)
    write_state(PUBLIC / 'report.json',result)
    print(json.dumps([{'seed':r['seed'],'free_oracle_nll':r['points'][0]['weighted_nll'],'controls':r['controls']} for r in rows],indent=2))


if __name__ == '__main__':
    main()
