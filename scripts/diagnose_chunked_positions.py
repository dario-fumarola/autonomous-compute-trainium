"""Prospective CPU-only position attribution after the unchanged main screen."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from compute_trainium.chunked_position_analysis import make_rows, summarize, compare, position_strata
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.readout_fixed_depth_causal import guard, indices_for_worker

ID = 'chunked-position-diagnostic-v1'
SCREEN = 'chunked-persistent-screen-v1'
PUBLIC = Path('results/chunked_persistent_workspace_20261029/position-diagnostic-v1')
FAMILIES = ('ordinary', 'generated', 'constant')


def prepare(root):
    source = root/SCREEN/'plan.json'
    main = json.loads(source.read_text())
    if (root/'runs'/(SCREEN+'-evaluation')).exists() or (root/SCREEN/'select').exists():
        raise AssertionError('Register before any main quality selection')
    files = dict(main['source_sha256'])
    for f in ('compute_trainium/chunked_position_analysis.py',
              'scripts/diagnose_chunked_positions.py', 'tests/test_chunked_position_analysis.py',
              'docs/CHUNKED_POSITION_DIAGNOSTIC_20261029.md'):
        files[f] = file_sha256(f)
    p = dict(identity=ID, registered_unix=time.time(), main_plan_sha256=file_sha256(source),
        source_sha256=files, absolute_deadline_unix=main['absolute_deadline_unix']+1200,
        seconds_limit=1200, workers=4, threads=8, bootstrap_seed=40514, bootstrap_replicates=2000,
        strata={k: dict(targets=len(v), positions=v) for k,v in position_strata().items()},
        causal_readout_condition='Both original main development gates pass; otherwise skip without replacing the failed result.',
        parameter_selection='Exactly the three independently selected main-screen endpoints; no reselection.',
        assigned_chips=0, optimizer_updates=0, protected_groups_scored=0, automatic_promotion=False,
        scope='Supplementary descriptive position analysis on reused DEV. Source-group intervals are conditional, unadjusted diagnostics, not new success gates or seed uncertainty. Blocking earlier latent keys is an out-of-distribution memory-dependence test, not proof of reasoning or a supported deployment budget.')
    (root/ID).mkdir(); PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json', p); write_state(PUBLIC/'plan.json', p)


def verify(root):
    p = json.loads((root/ID/'plan.json').read_text())
    if file_sha256(root/SCREEN/'plan.json') != p['main_plan_sha256']:
        raise AssertionError('Original scientific registration changed')
    if any(file_sha256(f) != h for f,h in p['source_sha256'].items()):
        raise AssertionError('Frozen diagnostic/source bytes changed')
    guard(root, p)
    main = json.loads((root/SCREEN/'plan.json').read_text())
    result = json.loads((root/SCREEN/'report.json').read_text())
    if result['status'] != 'COMPLETE_CHUNKED_PERSISTENT_SCREEN':
        raise AssertionError('Main fits, audits, selection and exact comparison must finish first')
    return p, main, result


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
    from compute_trainium.chunked_memory_intervention import readout
    from compute_trainium.model import ModelConfig

    p, main, result = verify(args.root); start = time.time()
    torch.set_num_threads(p['threads']); torch.set_num_interop_threads(1)
    admission = json.loads((args.root/ID/'start.json').read_text())
    if time.time()-admission['started_unix'] > p['seconds_limit']:
        raise TimeoutError('Original diagnostic duration')
    rank = args.worker; output = args.root/ID/'readout'
    evaluation = args.root/main['evaluation_relative']
    if any(file_sha256(evaluation/f) != h for f,h in main['evaluation_files'].items()):
        raise AssertionError('Original DEV bytes changed')
    tokens = np.load(evaluation/'broad-tokens.npy', mmap_mode='r')
    meta = json.loads((evaluation/'PRIVATE-broad-metadata.json').read_text())
    analysis = result['records']['compare']
    labels = (*FAMILIES, 'generated_memory_blocked') if analysis['both_development_gates_pass'] else FAMILIES
    scores, checks, hashes = {}, [], {}
    for label in labels:
        family = 'generated' if label == 'generated_memory_blocked' else label
        arm = main['arms'][analysis['selection'][family]['arm']]
        checkpoint = args.root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        h = file_sha256(checkpoint)
        if h != admission['checkpoint_sha256'][family]:
            raise AssertionError('Admitted audited checkpoint changed')
        hashes[label] = h
        model = ChunkedPersistentWorkspace(ModelConfig(**main['model_config'])).double().eval()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(saved['model']); del saved
        scores[label] = []
        with torch.inference_mode():
            if label == 'generated_memory_blocked':
                for domain in range(3):
                    guard(args.root, p)
                    i = domain*128+rank*2
                    x = torch.from_numpy(np.array(tokens[i:i+1,:-1], dtype=np.int64))
                    authentic = model.cached(x,128,1,'latent').log_softmax(-1)
                    reference = readout(model,x,128).log_softmax(-1)
                    blocked = readout(model,x,128,block_earlier_latents=True).log_softmax(-1)
                    error = float((authentic-reference).abs().max())
                    unaffected = float((blocked[:,:128]-reference[:,:128]).abs().max())
                    checks.append(dict(index=i, authentic_error=error, before_memory_read_error=unaffected))
                    if max(error,unaffected)>1e-8:
                        raise AssertionError('Independent intervention/reference causality check failed')
                    del authentic,reference,blocked
            for indices in indices_for_worker(rank,p['workers']):
                guard(args.root,p)
                if time.time()-admission['started_unix'] > p['seconds_limit']:
                    raise TimeoutError('Diagnostic duration')
                x = torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64))
                y = torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
                if label == 'generated_memory_blocked':
                    logits = readout(model,x,128,block_earlier_latents=True)
                else:
                    logits = model.cached(x,128,int(family!='ordinary'),'pause' if family=='constant' else 'latent')
                losses = F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                scores[label].extend(make_rows(losses.numpy(),meta,indices))
                del logits,losses
                write_state(output/f'worker{rank}-progress.json',dict(label=label,rows=len(scores[label]),wall_seconds=time.time()-start))
        del model
    write_state(output/f'PRIVATE-worker{rank}-scores.json',scores)
    write_state(output/f'worker{rank}-report.json',dict(status='COMPLETE_CHUNKED_POSITION_WORKER',
        labels=list(labels),checks=checks,checkpoint_sha256=hashes,wall_seconds=time.time()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root):
    p, main, result = verify(root)
    if managed_running():
        raise RuntimeError('Run after the Native campaign closes, without competing for host resources')
    out = root/ID; output = out/'readout'; output.mkdir(exist_ok=False)
    analysis = result['records']['compare']; hashes = {}
    for family in FAMILIES:
        armname = analysis['selection'][family]['arm']; arm = main['arms'][armname]
        checkpoint = root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        h = file_sha256(checkpoint)
        audit = result['records']['audit-'+armname]
        if audit['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS' or {r['checkpoint_sha256'] for r in audit['rows']}!={h}:
            raise AssertionError('Complete original audit required')
        hashes[family] = h
    start = time.time(); write_state(out/'start.json',dict(started_unix=start,
        main_result_sha256=file_sha256(root/SCREEN/'report.json'),checkpoint_sha256=hashes))
    jobs, logs = [], []; report = dict(status='STOPPED_CHUNKED_POSITION_DIAGNOSTIC')
    try:
        for rank in range(p['workers']):
            log = (output/f'worker{rank}.log').open('w'); logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.diagnose_chunked_positions',
                '--root',str(root),'--action','worker','--worker',str(rank)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if time.time()-start>p['seconds_limit']: raise TimeoutError('Fixed diagnostic cap')
            if any(j.poll() not in (None,0) for j in jobs): raise RuntimeError('CPU diagnostic worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs): raise RuntimeError('CPU diagnostic worker failed')
        receipts = [json.loads((output/f'worker{k}-report.json').read_text()) for k in range(p['workers'])]
        parts = [json.loads((output/f'PRIVATE-worker{k}-scores.json').read_text()) for k in range(p['workers'])]
        labels = receipts[0]['labels']
        if any(r['labels']!=labels or r['status']!='COMPLETE_CHUNKED_POSITION_WORKER' for r in receipts):
            raise AssertionError('Every worker must complete the same registered readouts')
        if sum(len(r['checks']) for r in receipts) != (12 if analysis['both_development_gates_pass'] else 0):
            raise AssertionError('Complete independent intervention checks required')
        by = {label:sorted([r for part in parts for r in part[label]],key=lambda r:r['index']) for label in labels}
        for family in FAMILIES:
            actual = summarize(by[family])
            expected = analysis['scores'][family]
            if not math.isclose(actual['broad_nll'],expected['broad_nll'],abs_tol=1e-8,rel_tol=0):
                raise AssertionError('Supplement must reproduce the unchanged original whole-panel result')
        kw = dict(seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'])
        comparisons = {f'generated_over_{family}':compare(by['generated'],by[family],**kw) for family in ('ordinary','constant')}
        if analysis['both_development_gates_pass']:
            comparisons['authentic_over_blocked_memory'] = compare(by['generated'],by['generated_memory_blocked'],**kw)
        report = dict(status='COMPLETE_CHUNKED_POSITION_DIAGNOSTIC',comparisons=comparisons,
            checkpoint_sha256=hashes,workers=receipts,original_main_decisions=analysis['decisions'],
            memory_intervention='completed' if analysis['both_development_gates_pass'] else 'skipped: original main gates did not both pass',
            private_score_sha256={f.name:file_sha256(f) for f in output.glob('PRIVATE-worker*-scores.json')})
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error)); raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        report.update(assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,
            terminal_alpha_spent=0,automatic_promotion=False,wall_seconds=time.time()-start,scope=p['scope'])
        write_state(out/'report.json',report);write_state(PUBLIC/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run','worker'],required=True);p.add_argument('--worker',type=int)
    a=p.parse_args();worker(a) if a.action=='worker' else (prepare if a.action=='prepare' else run)(a.root)
