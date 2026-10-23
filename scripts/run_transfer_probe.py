"""CPU feature extraction and one common, frozen residual-readout adaptation."""
import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compute_trainium.metrics import DocumentScore, paired_group_interval
from compute_trainium.training import file_sha256


def sample_group(s): return int(hashlib.sha256(('probe-inner-v1:'+s).encode()).hexdigest()[:16],16)%5


def readout_fit(features, original, labels, train, evaluation, lr, seed, snapshots=(0,64,256,512)):
    width=features.shape[1]
    x=F.normalize(features.float(),dim=-1)  # RMS normalization / sqrt(width).
    y=labels.long();base=original.float()
    weight=torch.zeros(4,width,requires_grad=True);bias=torch.zeros(4,requires_grad=True)
    opt=torch.optim.AdamW([weight,bias],lr=lr,weight_decay=.01)
    generator=torch.Generator().manual_seed(seed)
    history={}
    for step in range(513):
        if step in snapshots:
            with torch.no_grad():
                logits=base[evaluation]+F.linear(x[evaluation],weight,bias)
                losses=F.cross_entropy(logits,y[evaluation],reduction='none')
                history[step]={'losses':losses.numpy(),'correct':(logits.argmax(-1)==y[evaluation]).numpy()}
        if step==512:break
        batch=train[torch.randint(len(train),(64,),generator=generator)]
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(base[batch]+F.linear(x[batch],weight,bias),y[batch])
        loss.backward();opt.step()
    return history


def main():
    from transformers import AutoModelForCausalLM
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=12)
    p.add_argument('--maximum-seconds',type=int,default=10800)
    args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    start=time.monotonic()
    manifest=json.loads((args.data/'manifest.json').read_text())
    examples=json.loads((args.data/'examples.json').read_text())
    if file_sha256(args.data/'examples.json')!=manifest['examples_sha256']:raise ValueError('Frozen examples changed')
    registry=json.loads(args.registry.read_text())['models']
    names=['B0','scaffold_raw-seed11','scaffold_trace-seed11','scaffold_raw-seed12','scaffold_trace-seed12','mixed-seed11','mixed-seed12']
    spec={'protocol_sha256':file_sha256('docs/TRANSFER_PROBE_PROTOCOL.md'),'data_manifest_sha256':file_sha256(args.data/'manifest.json'),
          'examples_sha256':manifest['examples_sha256'],'models':names,'model_sha256':{n:registry[n]['native_checkpoint_sha256'] for n in names if n!='B0'},
          'readout_seeds':[101,202,303],'common_lr_candidates':[.003,.01,.03],
          'snapshots':[0,64,256,512],'cpu_threads':args.threads,'started_unix':time.time(),
          'max_seconds':args.maximum_seconds,'scope':'Frozen-backbone linear residual readout, not full fine-tuning'}
    (args.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    model,info=AutoModelForCausalLM.from_pretrained(args.base,local_files_only=True,trust_remote_code=False,
                      dtype=torch.float32,attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Non-exact base loading')
    model.eval();model.requires_grad_(False)
    rehearsal=[i for task in ['arc_challenge','race_high'] for i,d in
               list(enumerate(examples)) if d['task']==task and d['split']=='train']
    rehearsal=rehearsal[:8]+[i for i in rehearsal if examples[i]['task']=='race_high'][:8]
    rehearsal_start=time.monotonic();rehearsal_tokens=0
    with torch.inference_mode():
        for offset in range(0,len(rehearsal),8):
            chosen=rehearsal[offset:offset+8]
            lengths=[len(examples[i]['ids']) for i in chosen]
            ids=torch.zeros(len(chosen),max(lengths),dtype=torch.long);mask=torch.zeros_like(ids)
            for row,i in enumerate(chosen):
                ids[row,:lengths[row]]=torch.tensor(examples[i]['ids']);mask[row,:lengths[row]]=1
            model.model(input_ids=ids,attention_mask=mask,use_cache=False)
            rehearsal_tokens+=sum(lengths)
    rehearsal_seconds=time.monotonic()-rehearsal_start
    rate=rehearsal_tokens/rehearsal_seconds
    projected=2*len(names)*sum(len(d['ids']) for d in examples)/rate
    budget={'train_only_rehearsal_seconds':rehearsal_seconds,'train_only_tokens':rehearsal_tokens,
            'tokens_per_second':rate,'conservative_feature_seconds':projected,'maximum_seconds':args.maximum_seconds}
    (args.output/'measured-budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    if projected>args.maximum_seconds:raise ValueError('Measured transfer feature budget exceeds cap')
    label_ids=torch.tensor(manifest['answer_token_ids'])
    labels=torch.tensor([d['answer'] for d in examples])
    # Length sorting improves CPU utilization. Results retain original identity.
    order=sorted(range(len(examples)),key=lambda i:len(examples[i]['ids']))
    base_selection=None;records=[];stored={};total_forwards=total_tokens=0
    for name in names:
        if time.monotonic()-start>args.maximum_seconds:raise TimeoutError('Transfer CPU budget')
        if name!='B0':
            path=Path(registry[name]['native_model_file_on_existing_local_NVMe'])
            if file_sha256(path)!=spec['model_sha256'][name]:raise ValueError('Original model SHA changed')
            state=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
            model.load_state_dict(state['model'],strict=True);del state;gc.collect()
        features=torch.empty(len(examples),model.config.hidden_size,dtype=torch.float32)
        original=torch.empty(len(examples),4,dtype=torch.float32)
        first_check=None;model_start=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(order),8):
                chosen=order[offset:offset+8]
                lengths=torch.tensor([len(examples[i]['ids']) for i in chosen])
                ids=torch.zeros(len(chosen),int(lengths.max()),dtype=torch.long)
                mask=torch.zeros_like(ids)
                for row,i in enumerate(chosen):
                    ids[row,:lengths[row]]=torch.tensor(examples[i]['ids']);mask[row,:lengths[row]]=1
                hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                last=hidden[torch.arange(len(chosen)),lengths-1]
                if first_check is None:
                    single=model.model(input_ids=ids[:1,:lengths[0]],use_cache=False).last_hidden_state[0,-1]
                    first_check=float((single.float()-last[0].float()).abs().max())
                    if first_check>.0005:raise AssertionError('Batched/singleton FP32 features disagree')
                features[chosen]=last.float()
                original[chosen]=F.linear(last,model.lm_head.weight[label_ids]).float()
                total_forwards+=1;total_tokens+=int(lengths.sum())
                if offset%400==0:
                    print(json.dumps({'event':'features','model':name,'completed':offset+len(chosen),
                                      'model_seconds':time.monotonic()-model_start}),flush=True)
                if time.monotonic()-start>args.maximum_seconds:raise TimeoutError('Transfer CPU budget')
                available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
                if available<512*1024**3:raise RuntimeError('Protect host reserve')
        torch.save({'features':features,'original_logits':original,'labels':labels},args.output/(name+'-features.pt'))
        if name=='B0':
            selection=[]
            for lr in spec['common_lr_candidates']:
                task_values=[]
                for task in ['arc_challenge','race_high']:
                    train=torch.tensor([i for i,d in enumerate(examples) if d['task']==task and d['split']=='train' and sample_group(d['group'])!=0])
                    inner=torch.tensor([i for i,d in enumerate(examples) if d['task']==task and d['split']=='train' and sample_group(d['group'])==0])
                    values=readout_fit(features,original,labels,train,inner,lr,101,snapshots=(512,))
                    task_values.append(float(values[512]['losses'].mean()))
                selection.append({'lr':lr,'base_inner_train_nll':float(np.mean(task_values))})
            base_selection=min(selection,key=lambda d:d['base_inner_train_nll'])['lr']
            (args.output/'common-recipe.json').write_text(json.dumps({'selected_lr':base_selection,'selection':selection,
                'frozen_unix':time.time(),'criterion':'Untouched B0 TRAIN-only inner groups, equal task NLL'},indent=2)+'\n')
        model_record={'model':name,'feature_seconds':time.monotonic()-model_start,
                      'batch_singleton_max_abs_feature_difference':first_check,'curves':{}}
        model_scores={}
        for task in ['arc_challenge','race_high']:
            train=torch.tensor([i for i,d in enumerate(examples) if d['task']==task and d['split']=='train'])
            evaluation=torch.tensor([i for i,d in enumerate(examples) if d['task']==task and d['split']=='validation'])
            results=[readout_fit(features,original,labels,train,evaluation,base_selection,seed) for seed in spec['readout_seeds']]
            model_record['curves'][task]={str(step):{'mean_nll':float(np.mean([r[step]['losses'].mean() for r in results])),
                'mean_accuracy':float(np.mean([r[step]['correct'].mean() for r in results])),
                'seed_nll':[float(r[step]['losses'].mean()) for r in results],
                'seed_accuracy':[float(r[step]['correct'].mean()) for r in results],
                'examples':len(evaluation)} for step in spec['snapshots']}
            model_scores[task]=np.mean([r[512]['losses'] for r in results],axis=0)
        records.append(model_record);stored[name]=model_scores
        (args.output/'progress.json').write_text(json.dumps({'models':records,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
    contrasts={}
    def doc_scores(name):
        scores=[]
        for task in ['arc_challenge','race_high']:
            evaluation=[d for d in examples if d['task']==task and d['split']=='validation']
            scores.extend(DocumentScore(task,d['id'],d['group'],float(v),1) for d,v in zip(evaluation,stored[name][task],strict=True))
        return scores
    for seed in [11,12]:
        contrasts[str(seed)]=paired_group_interval(doc_scores(f'scaffold_raw-seed{seed}'),
            doc_scores(f'scaffold_trace-seed{seed}'),draws=2000,seed=17,weights={'arc_challenge':.5,'race_high':.5})
    report={'status':'COMPLETE_FROZEN_FEATURE_TRANSFER_SCREEN','models':records,'contrasts':contrasts,
            'common_lr':base_selection,'wall_seconds':time.monotonic()-start,'cpu_threads':args.threads,
            'backbone_batched_forwards':total_forwards,'prompt_tokens':total_tokens,
            'qualification_singleton_forwards':len(names)+len(rehearsal),
            'rehearsal':budget,
            'generated_examples':0,'accelerator_calls':0,'scope':spec['scope'],
            'warning':'Conditional on existing checkpoints; adaptation seeds are not new backbone replications. No full-backbone adaptation was tested.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'contrasts':contrasts,'wall_seconds':report['wall_seconds']}),flush=True)


if __name__=='__main__':main()
