"""Select each family independently; require useful gains against both controls."""
import argparse
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.analyze_depth_causal import summary
from scripts.analyze_ephemeral_tuning import read_scores,select
from scripts.compare_interleaved_tied import compare
from scripts.run_chunked_persistent_screen import FAMILIES


def verify(root,path):
    p=json.loads(path.read_text())
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):
        raise AssertionError('Frozen scientific sources changed')
    if time.time()>p['absolute_deadline_unix']:
        raise TimeoutError('Frozen screen deadline')
    return p


def decisions(comparisons,selections,margin):
    bracketed=all(s['bracketed'] for s in selections.values())
    return {k:bool(bracketed and v['comparison']['gain']>=margin and
                   v['comparison']['conditional_95_interval'][0]>0) for k,v in comparisons.items()}


def selection(root,p):
    folder=root/'runs'/(p['identity']+'-evaluation');scores={f:[] for f in FAMILIES}
    for name,a in p['arms'].items():
        audit=json.loads((root/p['identity']/('audit-'+name)/'report.json').read_text())
        if audit['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('All endpoints must be audited before quality selection')
        scores[a['family']].append(dict(arm=name,lr=a['lr'],**summary(read_scores(folder,name+'-main'))))
    chosen={f:select(rows,p['learning_rate_grid']) for f,rows in scores.items()}
    return dict(status='COMPLETE_CHUNKED_PERSISTENT_SELECTION',selection=chosen,all_scores=scores,
                selected_on='Reused broad DEV, each complete independent LR grid',automatic_promotion=False)


def comparison(root,p):
    out=root/p['identity'];sel=json.loads((out/'select/report.json').read_text())['selection']
    receipt=json.loads((out/'readout/report.json').read_text())
    if receipt['status']!='COMPLETE_CHUNKED_PERSISTENT_EXACT_READOUT' or receipt['selected_arms']!={f:s['arm'] for f,s in sel.items()}:
        raise AssertionError('Complete independently exact-scored selected models required')
    if any(file_sha256(out/'readout'/f)!=h for f,h in receipt['private_score_sha256'].items()):
        raise AssertionError('Exact private scores changed')
    rows=[r for f in sorted((out/'readout').glob('PRIVATE-worker*-scores.json')) for r in json.loads(f.read_text())]
    by={f:[r for r in rows if r['label']==f] for f in FAMILIES}
    kw=dict(seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'])
    comparisons={f'generated_over_{f}':compare(by['generated'],by[f],**kw) for f in ('ordinary','constant')}
    gates=decisions(comparisons,sel,p['practical_gain_nats']);resources={}
    for f,s in sel.items():
        arm=p['arms'][s['arm']];fit=json.loads((root/'runs'/arm['phase']/'report.json').read_text())
        resources[f]={k:fit[k] for k in ('parameters','updates','real_targets','nominal_training_matrix_flops','wall_seconds','mean_warm_update_seconds')}
        if fit['real_targets']!=arm['real_targets'] or fit['nominal_training_matrix_flops']!=arm['nominal_matrix_flops']:
            raise AssertionError('Complete registered exposure and work must match receipts')
    work=resources['generated']['nominal_training_matrix_flops']
    if resources['constant']['nominal_training_matrix_flops']!=work or resources['ordinary']['nominal_training_matrix_flops']<work:
        raise AssertionError('Constant work identical, ordinary never underfunded')
    return dict(status='COMPLETE_CHUNKED_PERSISTENT_ANALYSIS',selection=sel,
        scores={f:summary(v) for f,v in by.items()},comparisons=comparisons,decisions=gates,
        both_development_gates_pass=all(gates.values()),resources=resources,
        ordinary_work_ratio=resources['ordinary']['nominal_training_matrix_flops']/work,
        trained_seeds=1,protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
        uncertainty_scope='Conditional source-group intervals after independent LR selection on reused DEV; not training-seed uncertainty or confirmation.',
        scope=p['scope'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--action',choices=['select','compare'],required=True);a=p.parse_args();start=time.time()
    plan=verify(a.root,a.plan);result=(selection if a.action=='select' else comparison)(a.root,plan)
    result['wall_seconds']=time.time()-start;a.output.mkdir(exist_ok=False);write_state(a.output/'report.json',result)
