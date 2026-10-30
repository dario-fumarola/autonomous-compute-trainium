"""Descriptive operand-size breakdown of existing scores; no new inference."""
import json
from pathlib import Path
import resource
import time


def premise_ids(row):
    a,b=row['premise_tokens'];offset=row['window_start'];shift=int(offset==0)
    return row['tokens'][a-offset+shift:b-offset+shift]


def main(root):
    import numpy as np
    from tokenizers import Tokenizer
    from compute_trainium.direct_arithmetic_panel import expression_key
    from compute_trainium.source_arithmetic import check
    from compute_trainium.training import file_sha256
    from compute_trainium.queue import write_state
    start=time.time();panel=root/'primitive-arithmetic-panel-v1';scores=root/'primitive-arithmetic-readout-v1'
    out=root/'primitive-operand-size-summary-v4';public=Path('results/primitive_arithmetic_20261030/operand-size-v4')
    if out.exists() or public.exists():raise FileExistsError('Preserve every diagnostic')
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    plan=dict(registered_unix=start,absolute_deadline_unix=start+60,
        panel_sha256=file_sha256(panel/'PRIVATE-panel.json'),score_sha256={r:file_sha256(scores/('PRIVATE-'+r+'-scores.json')) for r in ('latent','ordinary','pause')},
        tokenizer_sha256=file_sha256(tokenizer),source_sha256=file_sha256(__file__),
        prior_attempt_sha256={str(v):file_sha256(root/f'primitive-operand-size-summary-v{v}/report.json') for v in (1,2,3)},
        bins=['both_abs_below10','both_abs_below100_remaining','some_abs_at_least100'],
        scope='Post-hoc descriptive partition of every previously scored elementary case by operand magnitude. No inference, source search, excluded case, selection, gate or research claim.')
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    t=Tokenizer.from_file(str(tokenizer));rows=json.loads((panel/'PRIVATE-panel.json').read_text())
    cells={r:{v['index']:v for v in json.loads((scores/('PRIVATE-'+r+'-scores.json')).read_text())} for r in plan['score_sha256']}
    memberships={};unresolved=[]
    for row in rows:
        text=t.decode(premise_ids(row),skip_special_tokens=False).strip()
        if expression_key(text)!=row['expression_key']:unresolved.append(row['index']);continue
        parsed=check(text)
        if parsed.operations!=1 or not parsed.literals:raise AssertionError('Original single-operation expression required')
        largest=max(abs(v) for v in parsed.literals)
        memberships[row['index']]=plan['bins'][0 if largest<10 else 1 if largest<100 else 2]
    if unresolved:raise AssertionError('Every original premise must reconstruct; no selective dropping')
    result={}
    for name in plan['bins']:
        indices=[i for i,b in memberships.items() if b==name]
        result[name]=dict(cases=len(indices),operators={o:sum(cells['latent'][i]['stratum']==o for i in indices) for o in ('add','subtract','multiply','divide')},
            models={r:dict(correct=sum(int(cells[r][i]['exact_prefix_correct']) for i in indices),
                answer_nll=float(np.mean([cells[r][i]['answer_nll'] for i in indices])) if indices else None,
                first_nll=float(np.mean([cells[r][i]['first_nll'] for i in indices])) if indices else None) for r in cells})
    if time.time()>plan['absolute_deadline_unix']:raise RuntimeError('Descriptive analysis time bound')
    report=dict(status='COMPLETE_SAVED_PRIMITIVE_OPERAND_BREAKDOWN',bins=result,cases=len(rows),
        all_premises_reconstruct=True,wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_inferences=0,assigned_chips=0,optimizer_updates=0,posthoc_descriptive=True,scope=plan['scope'])
    write_state(out/'report.json',report);write_state(public/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
