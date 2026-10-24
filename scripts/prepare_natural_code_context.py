"""Freeze authentic-code target positions without model scores or new text."""
import argparse
import ast
from collections import Counter,defaultdict
import hashlib
import json
import time
import warnings
from pathlib import Path

from tokenizers import Tokenizer

from compute_trainium.natural_code_context import unambiguous_local_uses,token_span,distance_band
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/'natural-code-context-v1';public=Path('results/natural_code_context_20261024')
    if out.exists():raise FileExistsError('Do not replace syntax-selected target panel')
    source=a.root/'variable-depth-fresh-v1/dev.json';tokenizer_path=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));docs=json.loads(source.read_text())
    config={'identity':'natural-code-context-v1','registered_unix':started,'source_sha256':file_sha256(source),
        'tokenizer_sha256':file_sha256(tokenizer_path),'split':'dev','cohort':'curve','domain':'code',
        'distance_bands':['129-256','257-512','513-900'],'full_positions':1024,'short_prefix_tokens':128,
        'maximum_identifier_tokens':16,'maximum_targets_per_band':60,'maximum_per_repository_band':8,
        'maximum_per_function_band':2,'minimum_per_band':10,'minimum_repositories_per_band':5,
        'selection':'Unique direct unconditional local assignment,dominating later same-scope identifier use. Reject rebinding/parameters/global/import/dynamic scope; prune nested scopes. Exact source token roundtrip,UTF8 AST alignment and whitespace-only token edge extension. Stable hash order,balanced bands,repository/function quotas.',
        'software_sha256':{f:file_sha256(f) for f in ['scripts/prepare_natural_code_context.py','compute_trainium/natural_code_context.py','tests/test_natural_code_context.py']},
        'model_scores_seen':False,'test_accessed':False,'scope':'Authentic unchanged held-out code tokens and lexical dependency positions; teacher-free syntax selection only. Development mechanism diagnostic,not general reasoning,synthetic programs or terminal confirmation.'}
    out.mkdir();public.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',config);write_state(public/'plan.json',config)
    counts=Counter();candidates=[]
    for d in sorted(docs,key=lambda d:d['id']):
        if (d['domain'],d['split'],d['cohort'])!=('code','dev','curve'):continue
        counts['eligible_code_documents']+=1;text=tokenizer.decode(d['tokens'],skip_special_tokens=False)
        encoded=tokenizer.encode(text,add_special_tokens=False)
        if encoded.ids!=d['tokens']:counts['non_roundtrip_documents']+=1;continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',SyntaxWarning);uses=unambiguous_local_uses(text)
        except (SyntaxError,ValueError,UnicodeError):counts['unparseable_python_documents']+=1;continue
        counts['parseable_python_documents']+=1
        for row in uses:
            counts['unambiguous_local_uses']+=1
            target=token_span(text,encoded.offsets,row['target_start'],row['target_end'])
            definition=token_span(text,encoded.offsets,row['definition_start'],row['definition_end'])
            if not target or not definition:counts['unaligned_token_edges']+=1;continue
            lo,hi=target;length=hi-lo
            if not 1<=length<=16:counts['identifier_token_length_exclusion']+=1;continue
            distance=lo-definition[1];band=distance_band(distance)
            if not band:counts['outside_distance_bands']+=1;continue
            full_start=max(0,lo-(1024-length+1))
            if definition[0]<full_start or lo-full_start<128:counts['definition_not_in_full_context']+=1;continue
            identity=hashlib.sha256((d['id']+':'+str(lo)+':'+str(hi)+':'+str(definition[0])).encode()).hexdigest()
            candidates.append({'id':identity,'document_id':d['id'],'group':d['group'],'function_line':row['function_line'],
                'target_start_token':lo,'target_end_token':hi,'definition_start_token':definition[0],
                'definition_end_token':definition[1],'full_start_token':full_start,'short_start_token':lo-128,
                'distance':distance,'band':band,'original_document_token_sha256':hashlib.sha256(json.dumps(d['tokens']).encode()).hexdigest()})
    chosen=[];by_band=Counter();by_repo=Counter();by_function=Counter()
    for row in sorted(candidates,key=lambda r:r['id']):
        band=row['band'];repo=(row['group'],band);function=(row['document_id'],row['function_line'],band)
        if by_band[band]>=60 or by_repo[repo]>=8 or by_function[function]>=2:continue
        chosen.append(row);by_band[band]+=1;by_repo[repo]+=1;by_function[function]+=1
    panel=out/'panel.json';write_state(panel,{'records':chosen})
    summary={}
    for band in config['distance_bands']:
        rows=[r for r in chosen if r['band']==band]
        summary[band]={'targets':len(rows),'repositories':len({r['group'] for r in rows}),
            'documents':len({r['document_id'] for r in rows}),'real_identifier_tokens':sum(r['target_end_token']-r['target_start_token'] for r in rows)}
    adequate=all(r['targets']>=10 and r['repositories']>=5 for r in summary.values())
    result={'status':'FROZEN_NATURAL_CODE_CONTEXT_PANEL' if adequate else 'INSUFFICIENT_PRESPECIFIED_NATURAL_CODE_COVERAGE',
        'bands':summary,'counts':dict(counts),'unscored_candidates':len(candidates),'selected_targets':len(chosen),
        'panel_sha256':file_sha256(panel),'plan_sha256':file_sha256(out/'plan.json'),'assigned_chips':0,
        'model_scores_seen':False,'optimizer_updates':0,'test_accessed':False,'wall_seconds':time.time()-started,
        'scope':config['scope']+' Insufficient coverage is reported,not repaired by generating code or selecting on outcomes.'}
    write_state(out/'report.json',result);write_state(public/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
