"""Fixed-weight CPU diagnostic: original human work versus no worked steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
os.environ.setdefault('USE_TORCH_XLA', '0')

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizers import Tokenizer
from compute_trainium.conclusion_generation import GreedyConclusion, correct_scalar
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN

IDENTITY = 'natural-chain-scaffold-readout-v1'
PUBLIC = Path('results/natural_chain_scaffold_readout_20261025')
SCOPE = ('Consumed DEV diagnostic, one pretrained seed, no learning. Original human work is an '
         'oracle intervention, not autonomous deployment input or a positive broad-text result. '
         'Evaluate the entire unchanged panel at both fixed views and prespecified exits; no subset selection.')


def register(root):
    parent = root / 'natural-chain-foundation-pilot-v1'
    spec = json.loads((parent / 'evaluation-plan.json').read_text())
    cases = {key: {**spec['cases'][key], 'depths': [depth]} for key, depth in
             [('one-bridge-work', 1), ('recurrent-bridge', 2)]}
    for row in cases.values():
        if file_sha256(root / row['checkpoint']) != row['checkpoint_sha256']:
            raise AssertionError('Completed frozen foundation fit changed')
    plan = dict(identity=IDENTITY, registered_unix=time.time(), cases=cases, model=spec['model'],
                panel=spec['panel'], panel_sha256=spec['panel_sha256'],
                parent_evaluation_plan_sha256=file_sha256(parent / 'evaluation-plan.json'),
                source_sha256={**spec['source_sha256'], str(Path(__file__).resolve().relative_to(Path.cwd())): file_sha256(__file__)},
                tokenizer_sha256=file_sha256(root.parent / 'data-v1/tokenizer.json'),
                views=['no_work', 'all_original_work'], maximum_new_tokens=24, seconds_limit=1200,
                assigned_chips=0, research_updates=0, test_accessed=False, terminal_alpha_spent=0,
                scope=SCOPE)
    out = root / IDENTITY
    out.mkdir(); PUBLIC.mkdir()
    write_state(out / 'plan.json', plan); write_state(PUBLIC / 'plan.json', plan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); out = a.root / IDENTITY
    if not (out / 'plan.json').exists(): register(a.root)
    if a.prepare_only: return
    if (out / 'report.json').exists(): raise FileExistsError('Preserve terminal diagnostic')
    plan = json.loads((out / 'plan.json').read_text()); started = time.time()
    result = dict(status='STOPPED_ORIGINAL_SCAFFOLD_DIAGNOSTIC', records=[], scope=SCOPE)
    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if time.time()-started > plan['seconds_limit'] or available < 512*1024**3 or min(shutil.disk_usage(out).free, shutil.disk_usage('/').free) < 35*1024**3:
            raise RuntimeError('CPU diagnostic cap/reserve')
    try:
        guard(); torch.set_num_threads(8); torch.set_num_interop_threads(1)
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()): raise AssertionError('Diagnostic source changed')
        panel_file = a.root / plan['panel']
        if file_sha256(panel_file) != plan['panel_sha256']: raise AssertionError('Original DEV changed')
        panel = json.loads(panel_file.read_text()); cfg = ParcaeCandidateConfig(**plan['model'])
        upstream = a.root / 'upstream-parcae-69284c1'
        if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PIN or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
            raise AssertionError('Clean pinned reference required')
        sys.path.insert(0, str(upstream))
        tokenizer_file = a.root.parent / 'data-v1/tokenizer.json'
        if file_sha256(tokenizer_file) != plan['tokenizer_sha256']: raise AssertionError('Original tokenizer changed')
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        all_records = []
        for key, case in plan['cases'].items():
            guard(); checkpoint = a.root / case['checkpoint']
            if file_sha256(checkpoint) != case['checkpoint_sha256']: raise AssertionError('Fitted state changed')
            saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            if saved['qualification_only'] or digest(dict(model=dict(sorted(saved['model'].items())), optimizer=saved['optimizer'])) != saved['state_digest']:
                raise AssertionError('Complete scientific state identity differs')
            model = make_reference(cfg, True).eval(); model.load_state_dict(saved['model'], strict=True); del saved
            before = digest(dict(sorted(model.state_dict().items()))); current = None
            model.initialize_state = lambda _: current
            for label in plan['views']:
                rows = []
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    for row in panel:
                        guard(); view = min(row['views'], key=lambda v: v['visible_steps']) if label == 'no_work' else max(row['views'], key=lambda v: v['visible_steps'])
                        if (label == 'no_work' and view['visible_steps'] != 0) or (label == 'all_original_work' and view['visible_steps'] < 1):
                            raise AssertionError('Every original problem must support both declared views')
                        prefix = view['prefix_ids']; seed = int(hashlib.sha256(json.dumps(prefix, separators=(',', ':')).encode()).hexdigest()[:16], 16) % (2**63)
                        current = initial_state(cfg, 0, seed)[:, :128]; run = GreedyConclusion(tuple(prefix), maximum_new_tokens=24)
                        while not run.done:
                            logits = model(torch.tensor(run.input_ids())[None], return_logits=True, num_steps_pair=torch.tensor([0, case['depths'][0]]))['logits']
                            run.advance(int(logits[0, run.prediction_position].argmax()))
                        text = tokenizer.decode(run.generated, skip_special_tokens=False)
                        rows.append(dict(document_id=row['id'], group=row['group'], visible_steps=view['visible_steps'], correct=correct_scalar(text, view['target'], run.ended), terminated=run.ended, generated_tokens=len(run.generated)))
                all_records.append(dict(case=key, view=label, rows=rows))
                result['records'].append(dict(case=key, view=label, depth=case['depths'][0], tasks=len(rows), correct=sum(r['correct'] for r in rows), terminated=sum(r['terminated'] for r in rows), generated_tokens=sum(r['generated_tokens'] for r in rows)))
                write_state(PUBLIC / 'progress.json', result)
            if digest(dict(sorted(model.state_dict().items()))) != before: raise AssertionError('Read-only weights changed')
            del model
        guard(); write_state(out / 'records.json', all_records)
        result.update(status='COMPLETE_ORIGINAL_SCAFFOLD_DIAGNOSTIC', records_sha256=file_sha256(out / 'records.json'), weights_unchanged=True)
    except BaseException as error:
        result['error'] = str(error); raise
    finally:
        result.update(wall_seconds=time.time()-started, assigned_chips=0, research_updates=0, test_accessed=False, terminal_alpha_spent=0, terminal_success=False, plan_sha256=file_sha256(out / 'plan.json'))
        write_state(out / 'report.json', result); write_state(PUBLIC / 'report.json', result)


if __name__ == '__main__':
    main()
