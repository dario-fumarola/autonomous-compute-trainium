"""Independent FP32 readout on frozen, previously consumed DEV only."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from compute_trainium.training import file_sha256,atomic_checkpoint
    from compute_trainium.scale_training_state import reconstruct_training_weights
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import initial_state
    from compute_trainium.data import document_windows
    from compute_trainium.metrics import DocumentScore,summarize
    from compute_trainium.execution_isolation import managed_running
    from scripts.train_scale_registered import lineage
    from scripts.audit_parcae_precise_training_point import make_reference
    from scripts.evaluate_low_depth_cpu import window_state_key
    from scripts.qualify_parcae_candidate_cpu import PIN
    from scripts.qualify_native import digest
    if a.output.exists():raise FileExistsError('Preserve all readouts')
    a.output.mkdir();plan=json.loads(a.plan.read_text());sha=file_sha256(a.plan);case=plan['cases'][a.case]
    source=Path(__file__).resolve().parents[1]
    if any(file_sha256(source/f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Frozen readout sources changed')
    if managed_running():raise RuntimeError('Dense independent scoring follows Native closure')
    audit=json.loads((a.root/case['audit_directory']/'report.json').read_text())
    if audit['status']!='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT' or audit['study_sha256']!=sha:
        raise ValueError('Completed exact endpoint numerical audit required before quality')
    if case['phase']=='gate':raise ValueError('Gate weights cannot receive scientific readouts')
    panel=a.root/plan['evaluation']['corpus']
    if (plan['evaluation']['scope']!='consumed_development_only'
            or file_sha256(panel)!=plan['evaluation']['corpus_sha256']):raise ValueError('Consumed DEV only')
    records=json.loads(panel.read_text());docs=[]
    for domain in ['web','math','code']:
        chosen=sorted([d for d in records if d['split']=='dev' and d['cohort']==case['cohort'] and d['domain']==domain],key=lambda d:d['id'])
        limit=plan['evaluation']['tune_documents_per_domain'] if case['phase']=='tune' else None
        docs.extend(chosen[:limit])
    if not docs:raise ValueError('Nonempty frozen cohort required')
    upstream=a.root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):raise AssertionError('Independent upstream changed')
    sys.path.insert(0,str(upstream));torch.set_num_threads(24)
    config=ParcaeCandidateConfig(**case['model']);recurrent=case['model_role']=='fixed_two';signature=lineage(sha,case)
    initial=torch.load(a.root/case['initializer'],map_location='cpu',mmap=True,weights_only=False)
    if file_sha256(a.root/case['initializer'])!=case['initializer_sha256']:raise AssertionError('Frozen C0 changed')
    weights,rows=reconstruct_training_weights(a.root/case['artifact_directory'],case['endpoint_updates'],signature,case['work_per_update'])
    state={**initial['model'],**weights};state['lm_head.weight']=state['transformer.wte.weight'];del rows,weights,initial;gc.collect()
    if recurrent:model=make_reference(config,True)
    else:
        from scripts.audit_ordinary8_native import make_reference as ordinary_reference
        model=ordinary_reference(config,False)
    model.load_state_dict(state,strict=True);model.eval();del state;gc.collect()
    model_digest=digest(dict(model.state_dict()));retained=a.root/case['retained_model']
    if retained.exists():raise FileExistsError('Retained model may not be overwritten')
    retained.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(retained.parent).free<35*1024**3+case['parameters']*4+256*1024**2:
        raise RuntimeError('Complete final-weight retention and protected reserve required')
    atomic_checkpoint(retained,dict(purpose='scale_retained_prediction_weights',phase=case['phase'],signature=signature,
        update=case['endpoint_updates'],eligible_for_science=True,model=model.state_dict(),model_digest=model_digest))
    readback=torch.load(retained,map_location='cpu',mmap=True,weights_only=False)
    if digest(readback['model'])!=model_digest:raise AssertionError('Exact retained final-weight readback failed')
    del readback
    retained_sha=file_sha256(retained);scores=[];windows=0;current=None
    if recurrent:model.initialize_state=lambda _:current
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3:raise RuntimeError('Independent scoring reserve')
        if time.time()>min(started+plan['readout_seconds_limit'],plan['absolute_deadline_unix']):raise TimeoutError('Original scoring/study cap')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for index,d in enumerate(docs):
            guard();total=0.;count=0
            for window,(tokens,mask) in enumerate(document_windows(d['tokens'],1024)):
                n=int(mask.sum());current=initial_state(config,window_state_key(d['id'],window),case['state_seed'])[:,:n] if recurrent else None
                kwargs={'num_steps_pair':torch.tensor([0,2])} if recurrent else {}
                logits=model(tokens[None,:n],return_logits=True,**kwargs)['logits'][0]
                if logits.shape!=(n,config.vocab_size) or not torch.isfinite(logits).all():raise AssertionError('Readout shape/finiteness failed')
                for start in range(0,n,128):
                    end=min(start+128,n);values=logits[start:end].double();targets=tokens[start+1:end+1]
                    total+=float((values.logsumexp(-1)-values.gather(-1,targets[:,None]).squeeze(-1)).sum())
                count+=n;windows+=1
            if count!=len(d['tokens']):raise AssertionError('Readout discarded source targets')
            scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
            if (index+1)%16==0:(a.output/'progress.json').write_text(json.dumps(dict(documents=index+1,total_documents=len(docs),elapsed_seconds=time.time()-started))+'\n')
    if digest(dict(model.state_dict()))!=model_digest:raise AssertionError('Independent readout changed weights')
    result=dict(status='COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT',case=a.case,phase=case['phase'],study_sha256=sha,
        seed=case['seed'],role=case['model_role'],depth=2 if recurrent else 1,parameters=case['parameters'],
        endpoint_updates=case['endpoint_updates'],real_targets=case['end_targets'],work=case['endpoint_updates']*case['work_per_update'],
        metrics=summarize(scores),documents=len(scores),windows=windows,cohort=case['cohort'],source_corpus_sha256=file_sha256(panel),
        retained_model_sha256=retained_sha,model_digest=model_digest,model_unchanged=True,
        assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
        scope='Independent FP32 prediction on explicitly consumed DEV. No fresh-source or training-seed confirmation.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'document-scores.json').write_text(json.dumps(dict(records=[dict(id=s.document,domain=s.domain,group=s.group,nll_sum=s.nll_sum,real_targets=s.tokens) for s in scores]))+'\n')
    print(json.dumps({'case':a.case,'nll':result['metrics']['weighted_nll'],'wall_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
