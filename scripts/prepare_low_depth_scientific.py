"""Fresh initializations and exact work tables; never launches a model fit."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict,replace
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_scientific_schedule import REFERENCE_BUDGET,update_work,matched_endpoint,source_endpoint,cooldown_start
from compute_trainium.parcae_training_candidate import poisson_tail
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN,reference_config,PlainTiedHead


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);a=p.parse_args()
    out=a.root/'parcae-low-depth-pilot-v1';out.mkdir(exist_ok=False)
    public=Path('results/parcae_low_depth_pilot_20261024');public.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.set_num_threads(8)
    if subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean pinned upstream initializer required')
    sys.path.insert(0,str(a.upstream))
    config=ParcaeCandidateConfig(vocab_size=32000)
    source=json.loads((a.root/'parcae-low-depth-qualification-v1/plan.json').read_text())
    files=['scripts/prepare_low_depth_scientific.py','compute_trainium/parcae_scientific_schedule.py',
           'compute_trainium/parcae_training_candidate.py','compute_trainium/parcae_work_accounting.py',
           'compute_trainium/parcae_candidate.py','scripts/qualify_parcae_candidate_cpu.py']
    plan={'identity':'parcae-low-depth-pilot-v1','registered_unix':started,'purpose':'natural_low_depth_scientific_preparation',
          'source_sha256':{f:file_sha256(f) for f in files},'upstream_commit':PIN,'model':asdict(config),
          'seeds':[11,1231],'tuning_seed':11,'pilot_seed':1231,'context':1024,'world':32,'batch':1,'accumulation':16,
          'routing_seed':271828,'state_seed_rule':'2009+model_seed','source_corpus_sha256':source['source_corpus_sha256'],
          'packed_manifest_sha256':source['packed_manifest_sha256'],'packing_fingerprint':source['packing_fingerprint'],
          'reference_budget_matmuls':REFERENCE_BUDGET,'tuning_updates':576,'equal_data_updates':1920,
          'cpu_seconds_limit':1200,'test_accessed':False,'optimizer_updates':0,'scientific_launch_authorized':False,
          'scope':'Fresh pinned random initialization for all arms/seeds and exact source/depth/work tables. Reuses only configuration/data provenance from qualification, never its weights. Final precision, tuning/launch budget and scientific gates remain separate.'}
    write_state(out/'preparation-plan.json',plan);write_state(public/'preparation-plan.json',plan)
    initials={};records=[]
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Scientific preparation host/disk reserve')
        if time.time()-started>1200:raise TimeoutError('Scientific CPU preparation twenty-minute cap')
    try:
        optimizer_work={};inventories={}
        with torch.device('meta'):
            for label,recurrent in [('ordinary',False),('family',True)]:
                model=ParcaeCandidate(config,recurrent=recurrent,initialize=False)
                optimizer_work[label]=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
                inventories[label]=sum(p.numel() for p in model.parameters())
        del model
        one=np.ones((1,16,32,2),dtype=np.uint16)
        minimum=min(int(update_work(config,one,recurrent=kind=='family',context=1024,world=32,
            optimizer_per_replica=optimizer_work[kind])[0]) for kind in inventories)
        max_updates=max(1921,REFERENCE_BUDGET//minimum+2)
        fixed=np.ones((max_updates,16,32,2),dtype=np.uint16)
        np.save(out/'fixed-depths.npy',fixed)
        random=fixed.copy()
        for update in range(max_updates):
            if update%64==0:guard()
            for micro in range(16):
                for rank in range(32):random[update,micro,rank]=poisson_tail(2,1,update*16+micro+1,rank)
        np.save(out/'poisson2-depths.npy',random)
        packed=PackedCorpus.from_mapped(a.root/'natural-append-6500m-v2/packed',1024,23,source['source_corpus_sha256'])
        if packed.fingerprint!=source['packing_fingerprint'] or max_updates*512>packed.windows:
            raise AssertionError('Insufficient or changed natural source provisioning')
        for arm,depths,kind in [('ordinary',fixed,'ordinary'),('fixed1',fixed,'family'),('poisson2',random,'family')]:
            guard();work=update_work(config,depths,recurrent=kind=='family',context=1024,world=32,
                                   optimizer_per_replica=optimizer_work[kind])
            np.save(out/(arm+'-work.npy'),work)
            endpoint=matched_endpoint(work,REFERENCE_BUDGET)
            endpoint.update(source_endpoint(packed.prefix_counts,endpoint['updates'],512))
            cooldown=cooldown_start(endpoint['updates'])
            endpoint['cooldown_start_update']=cooldown
            endpoint['cooldown_start_targets']=source_endpoint(packed.prefix_counts,cooldown,512)['real_targets']
            records.append({'arm':arm,'parameters':inventories[kind],
                'depth_table':'poisson2-depths.npy' if arm=='poisson2' else 'fixed-depths.npy',
                'depth_table_sha256':file_sha256(out/('poisson2-depths.npy' if arm=='poisson2' else 'fixed-depths.npy')),
                'work_table':arm+'-work.npy','work_table_sha256':file_sha256(out/(arm+'-work.npy')),
                'one_replica_optimizer_matmuls':optimizer_work[kind],
                'matched_work':endpoint,
                'tuning':{**source_endpoint(packed.prefix_counts,576,512),'updates':576,'cooldown_start_update':288,
                          'cooldown_start_targets':source_endpoint(packed.prefix_counts,288,512)['real_targets']},
                'equal_data':{**source_endpoint(packed.prefix_counts,1920,512),'updates':1920,'cooldown_start_update':960,
                          'cooldown_start_targets':source_endpoint(packed.prefix_counts,960,512)['real_targets']}})
        for seed in plan['seeds']:
            guard();torch.manual_seed(seed)
            family=replace(reference_config(None,config,True),mean_recurrence=2,mean_backprop_depth=1).construct_model().train()
            family_state=family.state_dict();ordinary=ParcaeCandidate(config,recurrent=False,initialize=False)
            mapped={}
            for name in ordinary.state_dict():
                source_name=name
                if name.startswith('transformer.h.'):
                    pieces=name.split('.');layer=int(pieces[2])
                    part,offset=('prelude',0) if layer<2 else ('core_block',2) if layer<4 else ('coda',4)
                    source_name='.'.join(['transformer',part,str(layer-offset),*pieces[3:]])
                mapped[name]=family_state[source_name]
            ordinary.load_state_dict(mapped)
            if not all(torch.equal(v,mapped[n]) for n,v in ordinary.state_dict().items()):
                raise AssertionError('Fresh shared backbone initialization differs')
            import parcae_lm.models.gpt.gpt as gpt
            gpt.LinearCrossEntropyLoss=PlainTiedHead
            independent=reference_config(None,config,False).construct_model()
            independent.load_state_dict(ordinary.state_dict(),strict=True)
            if digest(dict(sorted(independent.state_dict().items())))!=digest(dict(sorted(ordinary.state_dict().items()))):
                raise AssertionError('Fresh ordinary source mapping differs from upstream')
            for label,model in [('ordinary',ordinary),('family',family)]:
                name=f'initial-s{seed}-{label}.pt'
                atomic_checkpoint(out/name,{'model':model.state_dict(),'config':asdict(config),
                    'qualification_only':False,'seed':seed,'recurrent':label=='family','upstream_commit':PIN,
                    'provenance':'New pinned random initialization, never a qualification/trained checkpoint'})
                initials[f's{seed}-{label}']={'file':name,'sha256':file_sha256(out/name),
                    'model_digest':digest(dict(sorted(model.state_dict().items()))),'parameters':inventories[label]}
            del family,family_state,ordinary,independent,mapped;gc.collect()
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Scientific preparation source changed')
        plan.update(initial_states=initials,records=records,max_provisioned_updates=max_updates,
                    expected_depth=2+float(np.exp(-2)),realized_mean_depth=float(random[...,0].mean()),
                    maximum_depth=int(random[...,0].max()),all_shared_backbone_tensors_cloned=True)
        write_state(out/'preparation-plan.json',plan);write_state(public/'preparation-plan.json',plan)
        result={'status':'COMPLETE_LOW_DEPTH_SCIENTIFIC_PREPARATION','initial_states':initials,'records':records,
                'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,'scientific_launch_authorized':False,
                'wall_seconds_after_imports':time.time()-started,'scope':plan['scope']}
    except BaseException as e:
        result={'status':'STOPPED_LOW_DEPTH_SCIENTIFIC_PREPARATION','error':str(e),'records':records,
                'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,'wall_seconds_after_imports':time.time()-started}
        raise
    finally:
        write_state(out/'preparation-report.json',result);write_state(public/'preparation-report.json',result)


if __name__=='__main__':main()
