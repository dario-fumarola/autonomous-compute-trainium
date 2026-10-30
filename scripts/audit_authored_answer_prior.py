"""Train-only constant-answer calibration on the existing development panels."""
import argparse
from collections import Counter
import hashlib
import json
import mmap
from pathlib import Path
import resource
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_latent_exposure_sources import guard,read

ID='authored-answer-prior-v1'
PUBLIC=Path('results/primitive_arithmetic_20261030/answer-prior-v1')


def main(root):
    import numpy as np
    started=time.time();out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every diagnostic')
    pool=root/'calculation-document-pool-v1';pp=read(pool/'plan.json');pr=read(pool/'report.json')
    parent=root/'latent-exposure-source-audit-v1';ep=read(parent/'plan.json');er=read(parent/'report.json')
    elementary=root/'primitive-arithmetic-panel-v1';pe=read(elementary/'report.json')
    p=dict(identity=ID,registered_unix=started,absolute_deadline_unix=started+180,
        source_sha256={f:file_sha256(f) for f in ('scripts/audit_authored_answer_prior.py','scripts/audit_latent_exposure_sources.py','docs/AUTHORED_ANSWER_PRIOR_20261030.md')},
        pool_sha256=pr['files']['PRIVATE-pool.json'],parent_plan_sha256=file_sha256(parent/'plan.json'),
        panel_sha256={'elementary':pe['files']['PRIVATE-panel.json'],'multi':er['files']['PRIVATE-panel.json']},
        original_training_end=ep['endpoints']['one_billion']['virtual_position']+1,
        shard_sha256={str(i):er['files'][f'PRIVATE-documents-{i}.json'] for i in range(8)},
        scope=Path('docs/AUTHORED_ANSWER_PRIOR_20261030.md').read_text(),assigned_chips=0,
        optimizer_updates=0,model_inferences=0,fresh_reserves_scored=False)
    guard(root,p);out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    result=dict(status='STOPPED_AUTHORED_ANSWER_PRIOR')
    try:
        if file_sha256(pool/'PRIVATE-pool.json')!=p['pool_sha256']:raise AssertionError('Frozen pool changed')
        rows=read(pool/'PRIVATE-pool.json');wanted={r['id'] for r in rows};positions={}
        for i in range(8):
            path=parent/f'PRIVATE-documents-{i}.json'
            if file_sha256(path)!=p['shard_sha256'][str(i)]:raise AssertionError('TRAIN source shard changed')
            for r in read(path):
                if r['id'] in wanted:positions[r['id']]=r['start']
            guard(root,p)
        tokens=np.memmap(root/pp['source_relative']/'math.u16',dtype='<u2',mode='r');tokens._mmap.madvise(mmap.MADV_RANDOM)
        counts={'elementary':Counter(),'multi':Counter()}
        for i,row in enumerate(rows):
            start=positions[row['id']]
            for span in row['calculation_spans']:
                a,b=span['target_tokens'];first=start+1+a;last=start+1+b;window=((first-1)//512)*512
                if last>p['original_training_end'] or start+1+span['premise_tokens'][0]<window or last>window+513:continue
                answer=tuple(map(int,tokens[row['source_offset']+a:row['source_offset']+b]))
                counts['elementary' if span['operations']==1 else 'multi'][answer]+=1
            if i%512==0:tokens._mmap.madvise(mmap.MADV_DONTNEED);guard(root,p)
        choices={name:min(c,key=lambda k:(-c[k],k)) for name,c in counts.items()}
        # Freeze choices to disk before opening either evaluation answer file.
        write_state(out/'PRIVATE-train-choice.json',{name:dict(tokens=list(choice),frequency=counts[name][choice],
            training_occurrences=sum(counts[name].values())) for name,choice in choices.items()})
        tables={}
        for name,path in [('elementary',elementary/'PRIVATE-panel.json'),('multi',parent/'PRIVATE-panel.json')]:
            if file_sha256(path)!=p['panel_sha256'][name]:raise AssertionError('Frozen development panel changed')
            panel=read(path);by=Counter();correct=Counter();prefix=Counter();guess=choices[name]
            for row in panel:
                answer=tuple(row['tokens'][row['answer_start']:]);b=row['stratum']
                by[b]+=1;correct[b]+=int(answer==guess);prefix[b]+=int(len(guess)>=len(answer) and guess[:len(answer)]==answer)
            tables[name]=dict(training_occurrences=sum(counts[name].values()),distinct_training_answers=len(counts[name]),
                chosen_training_frequency=counts[name][guess],choice_sha256=hashlib.sha256(json.dumps(guess).encode()).hexdigest(),
                cases=len(panel),literal_complete_correct=sum(correct.values()),exact_prefix_correct=sum(prefix.values()),
                bins={b:dict(cases=by[b],literal_complete_correct=correct[b],exact_prefix_correct=prefix[b]) for b in sorted(by)})
        guard(root,p);result.update(status='COMPLETE_AUTHORED_ANSWER_PRIOR',panels=tables,
            selection_used_development_answers=False,post_hoc_development_calibration=True)
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            assigned_chips=0,optimizer_updates=0,model_inferences=0,fresh_reserves_scored=False,automatic_promotion=False,
            plan_sha256=file_sha256(out/'plan.json'))
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
