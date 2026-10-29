"""Separate descriptive position breakdown; original failed gates stay unmet."""
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

ID = 'chunked-completion-positions-v1'
SCREEN = 'chunked-control-completion-v1'
PUBLIC = Path('results/chunked_persistent_workspace_20261029/completion-positions-v1')
FAMILIES = ('ordinary', 'generated', 'constant')


def read(path):
    return json.loads(path.read_text())


def prepare(root, *, exploratory=False):
    source = root/SCREEN/'plan.json'
    main = read(source)
    evaluation = root/'runs'/(SCREEN+'-evaluation')
    if exploratory:
        if read(root/SCREEN/'report.json')['status']!='COMPLETE_CHUNKED_CONTROL_COMPLETION':
            raise AssertionError('Exploratory admission requires completed exact main comparison')
    elif (root/SCREEN/'select').exists() or (evaluation.exists() and any(evaluation.iterdir())):
        raise AssertionError('Register before any completion quality artifacts exist')
    files = dict(main['source_sha256'])
    for f in ('scripts/diagnose_chunked_completion_positions.py',
              'compute_trainium/chunked_position_analysis.py',
              'tests/test_chunked_completion_positions.py',
              'docs/CHUNKED_COMPLETION_POSITION_20261029.md'):
        files[f] = file_sha256(f)
    if exploratory:
        f='docs/CHUNKED_COMPLETION_POSITION_EXPLORATORY_20261029.md'; files[f]=file_sha256(f)
    plan = dict(identity=ID, registered_unix=time.time(),
        exploratory=exploratory, evaluation_process_started=evaluation.exists(),
        quality_artifacts_at_registration=(len(list(evaluation.iterdir())) if evaluation.exists() else 0),
        completion_plan_sha256=file_sha256(source), source_sha256=files,
        absolute_deadline_unix=main['absolute_deadline_unix']+1200,
        seconds_limit=1200, workers=4, threads=8, maximum_worker_rss_bytes=6*1024**3,
        bootstrap_seed=40514, bootstrap_replicates=2000,
        strata={k:dict(targets=len(v), positions=v) for k,v in position_strata().items()},
        assigned_chips=0, optimizer_updates=0, protected_groups_scored=0,
        automatic_promotion=False, memory_intervention=False,
        scope=('Exploratory decomposition after completed quality comparison. The earlier prospective draft was not admitted. ' if exploratory else 'New descriptive supplement before completion quality selection. ')+
        'Original screen stays STOPPED and original position diagnostic NOT ADMITTED. No new success gate, fit, selection, fresh sources, causal attribution or automatic promotion. Intervals condition on selected models and reused DEV.')
    (root/ID).mkdir(); PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json',plan); write_state(PUBLIC/'plan.json',plan)


def verify(root):
    p = read(root/ID/'plan.json'); guard(root,p)
    if file_sha256(root/SCREEN/'plan.json') != p['completion_plan_sha256']:
        raise AssertionError('Completion registration changed')
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):
        raise AssertionError('Frozen diagnostic sources changed')
    result = read(root/SCREEN/'report.json')
    if result['status'] != 'COMPLETE_CHUNKED_CONTROL_COMPLETION':
        raise AssertionError('Complete audited comparison required')
    return p, read(root/SCREEN/'readout-plan.json'), result


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
    from compute_trainium.model import ModelConfig
    p, main, result = verify(args.root); started = time.time()
    torch.set_num_threads(p['threads']); torch.set_num_interop_threads(1)
    admission = read(args.root/ID/'start.json'); output = args.root/ID/'readout'
    evaluation = args.root/main['evaluation_relative']
    if any(file_sha256(evaluation/f)!=h for f,h in main['evaluation_files'].items()):
        raise AssertionError('Original evaluation bytes changed')
    tokens = np.load(evaluation/'broad-tokens.npy',mmap_mode='r')
    meta = read(evaluation/'PRIVATE-broad-metadata.json')
    analysis = result['records']['compare']; scores = {}
    for family in FAMILIES:
        arm = main['arms'][analysis['selection'][family]['arm']]
        checkpoint = args.root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        if file_sha256(checkpoint)!=admission['checkpoint_sha256'][family]:
            raise AssertionError('Audited checkpoint changed')
        model = ChunkedPersistentWorkspace(ModelConfig(**main['model_config'])).double().eval()
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(saved['model']); del saved
        scores[family] = []
        with torch.inference_mode():
            for indices in indices_for_worker(args.worker,p['workers']):
                guard(args.root,p)
                if time.time()-admission['started_unix']>p['seconds_limit']:
                    raise TimeoutError('Fixed diagnostic duration')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['maximum_worker_rss_bytes']:
                    raise RuntimeError('Worker RSS reserve')
                x = torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64))
                y = torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
                logits = model.cached(x,128,int(family!='ordinary'),'pause' if family=='constant' else 'latent')
                losses = F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                scores[family].extend(make_rows(losses.numpy(),meta,indices))
                del logits,losses
                write_state(output/f'worker{args.worker}-progress.json',dict(family=family,rows=len(scores[family]),wall_seconds=time.time()-started))
        del model
    write_state(output/f'PRIVATE-worker{args.worker}-scores.json',scores)
    write_state(output/f'worker{args.worker}-report.json',dict(status='COMPLETE_COMPLETION_POSITION_WORKER',
        wall_seconds=time.time()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def aggregate(parts, expected, *, seed=40514, replicates=2000):
    by = {f:sorted([r for part in parts for r in part[f]],key=lambda r:r['index']) for f in FAMILIES}
    scores = {f:summarize(rows) for f,rows in by.items()}
    for family in FAMILIES:
        pairs = [(scores[family]['broad_nll'], expected[family]['broad_nll'])]
        pairs += [(scores[family]['domains'][d],expected[family]['domains'][d]) for d in ('web','math','code')]
        if any(not math.isclose(a,b,abs_tol=1e-8,rel_tol=0) for a,b in pairs):
            raise AssertionError('Every domain and broad score must reconstruct the exact comparison')
    comparisons = {f'{a}_over_{b}':compare(by[a],by[b],seed=seed,replicates=replicates)
                   for a,b in [('generated','ordinary'),('generated','constant'),('constant','ordinary')]}
    return dict(scores=scores,comparisons=comparisons)


def run(root):
    p,main,result = verify(root)
    if managed_running(): raise RuntimeError('Wait for clean Native closure')
    output = root/ID/'readout'; output.mkdir(exist_ok=False)
    analysis = result['records']['compare']; hashes = {}
    for f in FAMILIES:
        name = analysis['selection'][f]['arm']; arm = main['arms'][name]
        checkpoint = root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        h = file_sha256(checkpoint); audit = result['records']['audit-'+name]
        if audit['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS' or {r['checkpoint_sha256'] for r in audit['rows']}!={h}:
            raise AssertionError('Complete unchanged numerical audit required')
        hashes[f] = h
    started = time.time(); write_state(root/ID/'start.json',dict(started_unix=started,
        completion_result_sha256=file_sha256(root/SCREEN/'report.json'),checkpoint_sha256=hashes))
    jobs,logs = [],[]; report = dict(status='STOPPED_CHUNKED_COMPLETION_POSITIONS')
    try:
        for rank in range(p['workers']):
            log = (output/f'worker{rank}.log').open('w'); logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.diagnose_chunked_completion_positions',
                '--root',str(root),'--action','worker','--worker',str(rank)]+(['--exploratory'] if p['exploratory'] else []),stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if time.time()-started>p['seconds_limit']: raise TimeoutError('Fixed diagnostic duration')
            if any(j.poll() not in (None,0) for j in jobs): raise RuntimeError('CPU worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs): raise RuntimeError('CPU worker failed')
        receipts = [read(output/f'worker{k}-report.json') for k in range(p['workers'])]
        if any(r['status']!='COMPLETE_COMPLETION_POSITION_WORKER' for r in receipts):
            raise AssertionError('All workers must complete')
        scores = aggregate([read(output/f'PRIVATE-worker{k}-scores.json') for k in range(p['workers'])],
            analysis['scores'],seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'])
        report.update(status='COMPLETE_CHUNKED_COMPLETION_POSITIONS',**scores,workers=receipts,
            checkpoint_sha256=hashes,original_gate_status=analysis['original_gate_status'],
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
            automatic_promotion=False,terminal_alpha_spent=0,wall_seconds=time.time()-started,scope=p['scope'])
        write_state(root/ID/'report.json',report);write_state(PUBLIC/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run','worker'],required=True);p.add_argument('--worker',type=int)
    p.add_argument('--exploratory',action='store_true');a=p.parse_args()
    if a.exploratory:
        ID='chunked-completion-position-exploratory-v1'
        PUBLIC=PUBLIC.parent/'completion-position-exploratory-v1'
    if a.action=='worker':worker(a)
    elif a.action=='prepare':prepare(a.root,exploratory=a.exploratory)
    else:run(a.root)
