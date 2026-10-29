"""Descriptive selection after declared numerical failures; no gate repair."""
import argparse
from pathlib import Path
import time

from compute_trainium.queue import write_state
from scripts.analyze_chunked_persistent import verify, summary, comparison
from scripts.analyze_ephemeral_tuning import read_scores
from scripts.complete_chunked_controls import read


def select_available(rows,grid):
    if not rows or len({r['lr'] for r in rows})!=len(rows) or any(r['lr'] not in grid for r in rows):
        raise AssertionError('Unique registered finite completed candidates required')
    import math
    if any(not math.isfinite(r['broad_nll']) for r in rows):
        raise AssertionError('No invented infinite score for a failed fit')
    best=min(rows,key=lambda r:(r['broad_nll'],r['lr']))
    complete=sorted(r['lr'] for r in rows)==grid
    return dict(**best,bracketed=complete and 0<grid.index(best['lr'])<len(grid)-1,
                complete_numerical_grid=complete,selection_scope='Lowest NLL among complete audited original opportunities only')


def select(root,p):
    folder=root/'runs'/(p['identity']+'-evaluation');scores={f:[] for f in ('ordinary','generated','constant')}
    for arm,a in p['arms'].items():
        if read(root/p['identity']/('audit-'+arm)/'report.json')['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('No unaudited endpoint may be scored or selected')
        scores[a['family']].append(dict(arm=arm,lr=a['lr'],**summary(read_scores(folder,arm+'-main'))))
    return dict(status='COMPLETE_CHUNKED_COMPLETION_SELECTION',
        selection={f:select_available(rows,p['learning_rate_grid']) for f,rows in scores.items()},
        all_scores=scores,unavailable=p['unavailable_arms'],original_gate_status=p['original_gate_status'],
        automatic_promotion=False)


def compare(root,p):
    result=comparison(root,p)
    if any(result['decisions'].values()) or result['both_development_gates_pass']:
        raise AssertionError('Original incomplete generated grid cannot pass its gates')
    result.update(status='COMPLETE_CHUNKED_CONTROL_COMPARISON',unavailable=p['unavailable_arms'],
        original_gate_status=p['original_gate_status'],
        conclusion_scope='Descriptive reused-DEV comparison after a numerical tuning failure; original screen remains stopped. Point differences and conditional intervals do not repair the original advancement gate.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action',choices=['select','compare'],required=True);a=parser.parse_args();start=time.time()
    p=verify(a.root,a.plan);result=(select if a.action=='select' else compare)(a.root,p)
    result['wall_seconds']=time.time()-start;a.output.mkdir(exist_ok=False);write_state(a.output/'report.json',result)
