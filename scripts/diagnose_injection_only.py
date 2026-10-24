"""Small DEV-only same-weight boundary-injection ablation, not a trained control."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from compute_trainium.independent import IndependentModel
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def hidden_paths(model,tokens):
    n=tokens.shape[1];e=model.embedding(tokens)
    cos,sin=model.cos[:,:,:n].to(e.dtype),model.sin[:,:,:n].to(e.dtype)
    for i in range(model.prelude):e=model.blocks[i](e,cos,sin,None)
    one=e
    for i in range(model.prelude,model.prelude+model.core):one=model.blocks[i](one,cos,sin,None)
    injected=one+e;two=injected
    for i in range(model.prelude,model.prelude+model.core):two=model.blocks[i](two,cos,sin,None)
    result={}
    for name,state in [('one',one),('one_plus_injection',injected),('two',two)]:
        for i in range(model.prelude+model.core,model.config.layers):state=model.blocks[i](state,cos,sin,None)
        result[name]=model.norm(state)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(12);torch.set_num_interop_threads(1);started=time.time()
    corpus_sha=file_sha256(a.corpus);docs=sorted([d for d in json.loads(a.corpus.read_text())
        if d['split']=='dev' and d.get('cohort')=='curve' and len(d['tokens'])>=1024],key=lambda d:d['id'])
    chosen=[];groups=set()
    for domain in ['web','math','code']:
        selected=[]
        for doc in docs:
            if doc['domain']==domain and doc['group'] not in groups:
                selected.append(doc);groups.add(doc['group'])
            if len(selected)==4:break
        if len(selected)!=4:raise ValueError('Insufficient distinct-source long DEV documents')
        chosen+=selected
    private_selection=[{'id':d['id'],'group':d['group'],'domain':d['domain'],'targets':1024} for d in chosen]
    write_state(a.output/'private-selection.json',private_selection)
    spec={'identity':'injection-only-dev-v1','started_unix':started,'deadline_unix':a.deadline_unix,'seed':103,
        'corpus_sha256':corpus_sha,'selection_sha256':file_sha256(a.output/'private-selection.json'),
        'selection':'First four distinct-source DEV-curve documents/domain by content-hash order, length at least1024; score first1024 real tokens.',
        'models':['fixed','mixture'],'paths':['one','one_plus_injection','two'],'configured_cpu_threads':12,
        'source_sha256':{f:file_sha256(f) for f in ['scripts/diagnose_injection_only.py','compute_trainium/independent.py','compute_trainium/model.py']},
        'scope':'Exploratory first-predeclared-seed same-weight intervention on12 long natural DEV prefixes. Injection-only omits the second16-layer core but preserves the added prelude state. It was not trained in this form. It cannot establish how an independently trained cheap skip control would perform, replace the primary result, or support TEST/exit selection. No synthetic examples, fit or weight changes.'}
    write_state(a.output/'specification.json',spec);records=[]
    try:
        for arm in spec['models']:
            path=a.root/'runs'/f'vdepth-v1-{arm}-s103-u3840/checkpoint.pt';sha=file_sha256(path)
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
            if sig['dev_corpus_sha256']!=corpus_sha or saved['update']!=3840:raise ValueError('Wrong canonical checkpoint/panel')
            c=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
            model=IndependentModel(c,repetitions=1).eval();model.load_titan_state(saved['model']);del saved
            scores={name:[] for name in spec['paths']};begin=time.time()
            with torch.inference_mode():
                for index,doc in enumerate(chosen):
                    source=torch.tensor([1]+doc['tokens'][:1024],dtype=torch.long)
                    outputs=hidden_paths(model,source[None,:1024])
                    if index==0:
                        for depth,name in [(1,'one'),(2,'two')]:
                            model.repetitions=depth
                            if not torch.equal(outputs[name],model.hidden(source[None,:1024])):
                                raise AssertionError('Manual canonical path differs from independent reference')
                    for name,hidden in outputs.items():
                        total=0.
                        for first in range(0,1024,128):
                            end=first+128;logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double();target=source[first+1:end+1]
                            total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
                        scores[name].append(DocumentScore(doc['domain'],doc['id'],doc['group'],total,1024))
                    if time.time()>a.deadline_unix:raise TimeoutError('Bounded injection ablation deadline')
            if file_sha256(path)!=sha or file_sha256(a.corpus)!=corpus_sha:raise AssertionError('Read-only inputs changed')
            row={'arm':arm,'seed':103,'checkpoint_sha256':sha,'targets':12288,
                'metrics':{name:summarize(values) for name,values in scores.items()},'canonical_paths_bit_identical_to_reference':True,
                'active_seconds':time.time()-begin}
            records.append(row);write_state(a.output/'progress.json',{'records':records});print(json.dumps({'arm':arm,'nll':{k:v['weighted_nll'] for k,v in row['metrics'].items()}}),flush=True)
            del model,scores;gc.collect()
        write_state(a.output/'report.json',{'status':'COMPLETE_EXPLORATORY_INJECTION_ABLATION','specification':spec,
            'records':records,'wall_seconds':time.time()-started,'native_chips_assigned':0,'scope':spec['scope']})
    except Exception as error:
        write_state(a.output/'report.json',{'status':'STOPPED_INJECTION_ABLATION','error_type':type(error).__name__,'records':records,'wall_seconds':time.time()-started});raise


if __name__=='__main__':main()
