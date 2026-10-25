"""Post-hoc, model-free description of the fixed bridge TRAIN/DEV split."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import statistics

from compute_trainium.queue import write_state
from compute_trainium.source_arithmetic import check, normalize
from compute_trainium.training import file_sha256


def describe(view, tree_group):
    prefix = 'Evaluate the expression.\nExpression: '
    suffix = '\nAnswer:\n'
    if view['visible_steps'] or not view['prompt'].startswith(prefix) or not view['prompt'].endswith(suffix):
        raise ValueError('An unchanged no-work source view is required')
    expression = view['prompt'][len(prefix):-len(suffix)]
    normalized = normalize(expression)
    tree = ast.parse(normalized, mode='eval')
    literals = [ast.get_source_segment(normalized, n) for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    return dict(operations=check(expression).operations,
                operator_multiset=','.join(sorted(type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp))),
                operator_tree=tree_group, largest_literal_characters=max(map(len, literals)),
                answer_integer=check(view['target']).value.denominator == 1,
                answer_value=str(check(view['target']).value),
                answer_tokens=len(view['answer_ids']), prefix_tokens=len(view['prefix_ids']))


def summarize(rows):
    fields = ['operations', 'largest_literal_characters', 'answer_integer', 'answer_tokens']
    result = {key: dict(sorted(Counter(str(r[key]) for r in rows).items())) for key in fields}
    result.update(problems=len(rows), prefix_tokens=dict(min=min(r['prefix_tokens'] for r in rows),
                  median=statistics.median(r['prefix_tokens'] for r in rows), max=max(r['prefix_tokens'] for r in rows)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-views', type=Path, required=True)
    p.add_argument('--dev-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    training = json.loads(a.training_views.read_text())
    panel = json.loads(a.dev_panel.read_text())
    train = [describe(v, v['operation_tree_group']) for v in training if v['visible_steps'] == 0]
    dev = [describe(next(v for v in r['views'] if v['visible_steps'] == 0), r['operation_tree_group']) for r in panel]
    coverage = {}
    for key in ['operator_tree', 'operator_multiset', 'answer_value']:
        seen = {r[key] for r in train}
        coverage[key] = dict(dev_seen_in_train=sum(r[key] in seen for r in dev), dev_total=len(dev),
                             train_unique=len(seen), dev_unique=len({r[key] for r in dev}))
    write_state(a.output, dict(status='COMPLETE_POSTHOC_SOURCE_SPLIT_DESCRIPTION',
                training_views_sha256=file_sha256(a.training_views), dev_panel_sha256=file_sha256(a.dev_panel),
                source_sha256=file_sha256(__file__), train=summarize(train), dev=summarize(dev), coverage=coverage,
                assigned_chips=0, research_updates=0, test_accessed=False,
                scope='Post-hoc aggregate description after the original pilot failed. No model scores, new split, '
                      'task selection, synthetic labels or changes to the running foundation-history comparison. '
                      'Shared operator structure or answer values do not imply duplicate mathematical problems.'))


if __name__ == '__main__':
    main()
