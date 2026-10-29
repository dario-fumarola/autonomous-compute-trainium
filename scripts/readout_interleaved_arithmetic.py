"""Frozen existing arithmetic/code DEV cases; no new fit or case selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.readout_interleaved_bracket_type import checkpoint

ID = 'interleaved-arithmetic-dev-v1'
PUBLIC = Path('results/interleaved_workspace_20261029/arithmetic-dev-v1')
KINDS = ('math_conclusion', 'math_intermediate_exploratory', 'code_binding')
LABELS = ('latent', 'ordinary', 'pause')


def view(x, y, mask, condition):
    import numpy as np
    pos = np.flatnonzero(mask).tolist()
    if not pos or pos != list(range(pos[0], pos[-1]+1)) or condition not in ('full', 'last32'):
        raise ValueError('Contiguous authentic answer and declared context required')
    start = 0 if condition == 'full' else max(0, pos[0]-31)
    return x[start:pos[-1]+1], y[pos], [p-start for p in pos]


def analysis(rows, seed=40303, replicates=2000):
    from scripts.analyze_probability_screen import paired, score
    cells = {(label, condition, kind): [r for r in rows if (r['label'],r['condition'],r['kind']) == (label,condition,kind)]
             for label in LABELS for condition in ('full', 'last32') for kind in KINDS}
    for kind in KINDS:
        expected = {r['id'] for r in cells[('latent','full',kind)]}
        if not expected:
            raise AssertionError('Every frozen task family must remain present')
        for label in LABELS:
            for condition in ('full','last32'):
                own = cells[(label,condition,kind)]
                if len(own) != len(expected) or {r['id'] for r in own} != expected:
                    raise AssertionError('Complete paired grid without dropped cases required')
    means = {label:{condition:{} for condition in ('full','last32')} for label in LABELS}
    for (label,condition,kind), values in cells.items():
        means[label][condition][kind] = dict(answer_nll=score(values), first_target_nll=score(values,'first_loss'),
            token_accuracy=sum(r['correct_tokens'] for r in values)/sum(r['targets'] for r in values),
            all_answer_tokens_correct=sum(r['whole_correct'] for r in values)/len(values),
            cases=len(values), groups=len({r['group'] for r in values}), targets=sum(r['targets'] for r in values))
    comparisons = {}
    for kind in KINDS:
        own = {}
        for control in ('ordinary','pause'):
            own['full_vs_'+control] = paired(cells[(control,'full',kind)],cells[('latent','full',kind)],seed=seed,replicates=replicates)
            own['first_target_vs_'+control] = paired(cells[(control,'full',kind)],cells[('latent','full',kind)],seed=seed,replicates=replicates,field='first_loss')
        for label in LABELS:
            own['context_utility_'+label] = paired(cells[(label,'last32',kind)],cells[(label,'full',kind)],seed=seed,replicates=replicates)
        comparisons[kind] = own
    gates = {label: comparisons['math_conclusion']['full_vs_'+label]['gain'] >= .03 and
             comparisons['math_conclusion']['full_vs_'+label]['conditional_95_interval'][0] > 0
             for label in ('ordinary','pause')}
    return dict(means=means, comparisons=comparisons, conclusion_gates=gates,
                note='Conditional source-group intervals for one selected development seed. '
                     'All-answer accuracy uses teacher-forced argmax, not free-form solving or termination.')


def guard(root, plan):
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if (time.time() > plan['absolute_deadline_unix'] or available < 512*1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3
            or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free) < 35*1024**3):
        raise RuntimeError('Frozen CPU readout resource cap')


def prepare(root):
    panel = root/'probability-workspace-evaluation-v1'
    receipt = json.loads((panel/'report.json').read_text())
    expected = dict(math_conclusion=134, math_intermediate_exploratory=110, code_binding=9)
    if receipt['status'] != 'FROZEN_PROBABILITY_EVALUATION' or receipt['dependency_counts'] != expected:
        raise AssertionError('Use the complete existing 253-case panel')
    files = {n:h for n,h in receipt['files'].items() if 'dependency' in n}
    if any(file_sha256(panel/n) != h for n,h in files.items()):
        raise AssertionError('Original authentic source panel changed')
    meta = json.loads((panel/'PRIVATE-dependency-metadata.json').read_text())
    if dict(Counter(r['kind'] for r in meta)) != expected or len({r['id'] for r in meta}) != 253:
        raise AssertionError('Every original case required exactly once')
    base = Path('results/interleaved_workspace_20261029')
    mixed = json.loads((base/'mixed-round-tuning-v1/analysis.json').read_text())['selection']['arm']
    tied = json.loads((base/'tied-ordinary-tuning-v1/analysis.json').read_text())['selection']['arm']
    points = dict(latent=checkpoint(root,'interleaved-mixed-round-tuning-v1',mixed),
                  ordinary=checkpoint(root,'interleaved-tied-ordinary-tuning-v1',tied),
                  pause=checkpoint(root,'interleaved-lr-tuning-v1','pause-lr1'))
    for point in points.values():
        fit = json.loads((root/point['plan_relative']).read_text())
        exclusions = json.loads((root/fit['data_relative']/'plan.json').read_text())['excluded_group_files']
        if exclusions['probability-workspace-evaluation-v1/PRIVATE-dependency-metadata.json'] != files['PRIVATE-dependency-metadata.json']:
            raise AssertionError('Every selected fit must exclude these exact source groups')
        if fit['seed'] != 9901:
            raise AssertionError('Original selected seed only; new replication readout stays frozen')
    sources = [str(f) for f in Path('compute_trainium').glob('*.py')]
    sources += ['scripts/readout_interleaved_arithmetic.py', 'scripts/analyze_probability_screen.py',
                'docs/INTERLEAVED_ARITHMETIC_DEV_20261029.md', 'tests/test_interleaved_arithmetic.py',
                'tests/test_interleaved_selected_readout.py']
    now = time.time()
    plan = dict(identity=ID, registered_unix=now, absolute_deadline_unix=now+3600,
                source_sha256={f:file_sha256(f) for f in sources}, points=points,
                panel_relative=panel.name, panel_report_sha256=file_sha256(panel/'report.json'), panel_files=files,
                primary='Math-conclusion whole-answer NLL; all 134 authored answers; both fixed controls.',
                practical_gain=.03, bootstrap_seed=40303, bootstrap_replicates=2000,
                assigned_chips=0, optimizer_updates=0, protected_groups_scored=0, terminal_alpha_spent=0,
                scope='One selected seed, existing reused DEV; authored arithmetic with original work supplied. '
                      'No new training, case selection, prompt search, unsupported exit or fresh reserve. '
                      'A positive is arithmetic prediction evidence, not autonomous allocation, general reasoning or efficiency.')
    guard(root, plan)
    out = root/ID
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan)
    write_state(PUBLIC/'plan.json', plan)


def run(root):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from compute_trainium.interleaved_selected_readout import cached_selected_logits
    out = root/ID
    plan = json.loads((out/'plan.json').read_text())
    start = time.time()
    result = dict(status='RUNNING_INTERLEAVED_ARITHMETIC_DEV', assigned_chips=0, optimizer_updates=0,
                  protected_groups_scored=0, terminal_alpha_spent=0)
    rows, checks = [], []
    try:
        if any(file_sha256(f) != h for f,h in plan['source_sha256'].items()):
            raise AssertionError('Frozen CPU source changed')
        panel = root/plan['panel_relative']
        if file_sha256(panel/'report.json') != plan['panel_report_sha256'] or any(file_sha256(panel/n) != h for n,h in plan['panel_files'].items()):
            raise AssertionError('Frozen panel changed')
        xs = np.load(panel/'dependency-inputs.npy')
        ys = np.load(panel/'dependency-targets.npy')
        ms = np.load(panel/'dependency-mask.npy')
        meta = json.loads((panel/'PRIVATE-dependency-metadata.json').read_text())
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        for label in LABELS:
            guard(root, plan)
            point = plan['points'][label]
            if file_sha256(root/point['plan_relative']) != point['plan_sha256'] or file_sha256(root/point['checkpoint_relative']) != point['checkpoint_sha256']:
                raise AssertionError('Audited endpoint changed')
            fit = json.loads((root/point['plan_relative']).read_text())
            model = (tied_model(fit) if label == 'ordinary' else make_model(fit)).double().eval()
            saved = torch.load(root/point['checkpoint_relative'], map_location='cpu',weights_only=False,mmap=True)
            model.load_state_dict(saved['model'])
            del saved
            versions = tuple(p._version for p in model.parameters())
            with torch.inference_mode():
                if label == 'latent':
                    for i in range(6):
                        x = torch.from_numpy(np.array(xs[i:i+1,:32],dtype=np.int64))
                        a = cached_selected_logits(model,x,[28,31])
                        b = model.sequential(x)[:,[28,31]]
                        error = float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())
                        checks.append(dict(index=i,maximum_log_probability_error=error))
                        if error > 1e-8:
                            raise AssertionError('Actual-checkpoint selected/full-prefix parity')
                for condition in ('full','last32'):
                    for i,item in enumerate(meta):
                        guard(root, plan)
                        xx,yy,positions = view(xs[i],ys[i],ms[i],condition)
                        x = torch.from_numpy(np.array(xx[None],dtype=np.int64))
                        y = torch.from_numpy(np.array(yy,dtype=np.int64))
                        logits = (cached_selected_logits(model,x,positions)[0] if label == 'latent'
                                  else model(x,0,label)[0,positions])
                        losses = F.cross_entropy(logits,y,reduction='none')
                        correct = logits.argmax(-1) == y
                        if not torch.isfinite(losses).all():
                            raise FloatingPointError('Nonfinite answer loss')
                        rows.append(dict(label=label,condition=condition,id=item['id'],group=item['group'],kind=item['kind'],
                                         cohort=item['cohort'],targets=len(y),loss_sum=float(losses.sum()),first_loss=float(losses[0]),
                                         correct_tokens=int(correct.sum()),whole_correct=int(correct.all())))
                        if i % 16 == 0:
                            write_state(out/'progress.json',dict(label=label,condition=condition,completed_cases=i+1,seconds=time.time()-start))
                    write_state(out/'PRIVATE-scores.json',rows)
            if versions != tuple(p._version for p in model.parameters()):
                raise AssertionError('Readout changed trained parameters')
            del model
        result.update(status='COMPLETE_INTERLEAVED_ARITHMETIC_DEV', analysis=analysis(rows,plan['bootstrap_seed'],plan['bootstrap_replicates']),
                      checks=checks,trained_seeds=1,scope=plan['scope'],automatic_promotion=False)
    except BaseException as error:
        result.update(status='STOPPED_INTERLEAVED_ARITHMETIC_DEV',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_state(out/'report.json',result)
        write_state(PUBLIC/'report.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--action',choices=['prepare','run'],required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.root)
