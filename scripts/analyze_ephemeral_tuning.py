"""Select the frozen three-LR control and compare all original paired endpoints."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.analyze_probability_screen import score
from scripts.compare_interleaved_tied import compare, validate_rows


def select(scores, grid):
    candidates = sorted(scores, key=lambda r: r['lr'])
    if [r['lr'] for r in candidates] != grid:
        raise AssertionError('Exactly the complete frozen LR grid required')
    best = min(candidates, key=lambda r: (r['broad_nll'], r['lr']))
    return dict(**best, bracketed=0 < candidates.index(best) < len(candidates)-1)


def read_scores(folder, label):
    if json.loads((folder/'report.json').read_text())['status'] != 'COMPLETE_INTERLEAVED_SCREEN_EVALUATION':
        raise AssertionError('Complete audited hardware readout required')
    rows = [r for rank in range(64) for r in json.loads((folder/f'PRIVATE-rank{rank}-scores.json').read_text())
            if r['label'] == label and r['panel'] == 'broad']
    validate_rows(rows)
    return rows


def main(args):
    start = time.time()
    p = json.loads(args.plan.read_text())
    if any(file_sha256(f) != h for f, h in p['source_sha256'].items()):
        raise AssertionError('Frozen analysis source changed')
    if any(file_sha256(args.root/f) != h for f, h in p['comparison_artifact_sha256'].items()):
        raise AssertionError('Registered comparison artifacts changed')
    folder = args.root/'runs'/(p['identity']+'-evaluation')
    all_rows, scores = {}, []
    for name, arm in p['arms'].items():
        audit = args.root/p['identity']/('audit-'+name)/'report.json'
        if json.loads(audit.read_text())['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('Every mature audit must pass')
        rows = read_scores(folder, name+'-main')
        all_rows[name] = rows
        domains = {d: score([r for r in rows if r['domain'] == d]) for d in ('web', 'math', 'code')}
        scores.append(dict(arm=name, lr=arm['lr'], domains=domains,
                           broad_nll=sum(w*domains[d] for d, w in zip(('web', 'math', 'code'), (.6, .2, .2)))))
    selected = select(scores, p['learning_rate_grid'])
    ephemeral = all_rows[selected['arm']]
    persistent = [r for f in sorted((args.root/'interleaved-mixed-sequential-dev-v1').glob('PRIVATE-worker*-scores.json'))
                  for r in json.loads(f.read_text())]
    tied = json.loads((args.root/'interleaved-tied-ordinary-tuning-v1/analysis/report.json').read_text())['selection']
    ordinary = read_scores(args.root/'runs/interleaved-tied-ordinary-tuning-v1-evaluation', tied['arm']+'-main')
    old = json.loads(Path('results/interleaved_workspace_20261029/tuning-v1/analysis.json').read_text())
    pause = read_scores(args.root/'runs/interleaved-lr-tuning-v1-evaluation', old['selection']['pause']['arm']+'-main')
    kwargs = dict(seed=p['bootstrap_seed'], replicates=p['bootstrap_replicates'])
    comparisons = dict(persistent_over_ephemeral=compare(persistent, ephemeral, **kwargs),
                       ephemeral_over_ordinary=compare(ephemeral, ordinary, **kwargs),
                       ephemeral_over_pause=compare(ephemeral, pause, **kwargs))
    practical = p['practical_gain_nats']
    def passes(label):
        c = comparisons[label]['comparison']
        return selected['bracketed'] and c['gain'] >= practical and c['conditional_95_interval'][0] > 0
    resources = {}
    for label, plan_path, arm_name in [
            ('ephemeral', args.plan, selected['arm']),
            ('persistent', args.root/'interleaved-mixed-round-tuning-v1/plan.json',
             json.loads((args.root/'interleaved-mixed-sequential-dev-v1/plan.json').read_text())['selected_arm']),
            ('ordinary', args.root/'interleaved-tied-ordinary-tuning-v1/plan.json', tied['arm'])]:
        plan = json.loads(plan_path.read_text())
        arm = plan['arms'][arm_name]
        fit = json.loads((args.root/'runs'/arm['phase']/'report.json').read_text())
        if arm['real_targets'] != p['arms'][selected['arm']]['real_targets']:
            raise AssertionError('Equal complete authentic exposure required')
        resources[label] = dict(parameters=fit['parameters'], real_targets=fit['real_targets'],
                                nominal_training_matrix_flops=fit['nominal_training_matrix_flops'],
                                worker_fit_wall_seconds=fit['wall_seconds'])
    result = dict(status='COMPLETE_EPHEMERAL_TRAINED_COMPARISON', selection=selected,
                  scores=scores, comparisons=comparisons, resources=resources,
                  persistent_memory_development_screen=passes('persistent_over_ephemeral'),
                  ephemeral_prediction_development_screen=passes('ephemeral_over_ordinary') and passes('ephemeral_over_pause'),
                  trained_seeds=1, assigned_chips=0, optimizer_updates=0,
                  protected_groups_scored=0, terminal_alpha_spent=0, automatic_promotion=False,
                  wall_seconds=time.time()-start,
                  interpretation='Positive gains favor the first-named model. Every source interval is conditional on these selected development fits. Failure to find a difference does not establish equivalence. Training changes memory access and optimization work jointly; this is not a same-checkpoint content intervention. No general reasoning or efficiency inference.',
                  scope=p['scope'])
    args.output.mkdir(exist_ok=False)
    write_state(args.output/'report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
