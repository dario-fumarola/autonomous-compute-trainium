"""CPU-only full natural-source counters for a finite one/fixed-two NTP curve."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('USE_TORCH_XLA','0')

import numpy as np
import torch
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_data import count_future_anchors
from compute_trainium.future_span_scientific_state import matrix_work
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.qualify_native import digest

IDENTITY='fixed-two-ntp-main-preparation-v1'
PUBLIC=Path('results/fixed_two_small_core_main_preparation_20261025')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/IDENTITY;out.mkdir();PUBLIC.mkdir()
    parent=a.root/'future-span-study-preparation-v1';old=json.loads((parent/'manifest.json').read_text())
    files=['scripts/prepare_fixed_two_ntp_main.py','compute_trainium/future_span_data.py','compute_trainium/future_span_scientific_state.py','compute_trainium/parcae_candidate.py','compute_trainium/parcae_scientific_schedule.py','compute_trainium/training.py','docs/FIXED_TWO_SMALL_CORE_MAIN_20261025.md']
    registration=dict(identity=IDENTITY,registered_unix=started,parent_manifest_sha256=file_sha256(parent/'manifest.json'),source_sha256={f:file_sha256(f) for f in files},rows=5121,seed=5231,scientific_launch_allowed=False,assigned_chips=0,research_updates=0,seconds_limit=600,test_accessed=False,terminal_alpha_spent=0,scope='CPU-only prospective data/schedule/work preparation. Original random initializer and unchanged authentic-source packing;no model fit,quality readout or scientific admission.')
    write_state(out/'registration.json',registration);write_state(PUBLIC/'registration.json',registration)
    result=dict(status='STOPPED_FIXED_TWO_NTP_MAIN_PREPARATION')
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        require_inode_reserve([out,Path('/')])
        if time.time()-started>600 or available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('CPU curve preparation cap/reserve')
    try:
        guard();torch.set_num_threads(4)
        if any(file_sha256(f)!=h for f,h in old['source_sha256'].items()):raise AssertionError('Original qualified preparation changed')
        cfg=ParcaeCandidateConfig(**old['model']);span=FutureSpanConfig(**old['future_config']);data=old['data']
        if file_sha256(a.root/data['packed']/'manifest.json')!=data['manifest_sha256']:raise AssertionError('Frozen natural packing changed')
        packed=PackedCorpus.from_mapped(a.root/data['packed'],1024,23,data['corpus_sha256'])
        if packed.fingerprint!=data['packing_fingerprint'] or packed.windows<registration['rows']*512:raise ValueError('Adequate unchanged original source required;no wrapping')
        counts,by_domain=count_future_anchors(packed.domains.numpy(),span,registration['rows']*512)
        prefix=np.concatenate([np.zeros(1,dtype=np.int64),counts.cumsum(dtype=np.int64)])
        original_prefix=np.load(parent/old['auxiliary_data']['prefix_file'],mmap_mode='r')
        if file_sha256(parent/old['auxiliary_data']['prefix_file'])!=old['auxiliary_data']['prefix_sha256'] or not np.array_equal(prefix[:len(original_prefix)],original_prefix):raise AssertionError('Append-compatible original anchor counts differ')
        np.save(out/'anchor-counts.npy',counts);np.save(out/'anchor-prefix.npy',prefix)
        if not np.array_equal(np.load(out/'anchor-prefix.npy'),prefix):raise AssertionError('Complete anchor-prefix readback differs')
        tables={};work_by_arm={}
        for arm,n in [('one',1),('fixed',2)]:
            guard();depth=np.full((registration['rows'],16,32,2),n,dtype=np.int16)
            work=matrix_work(cfg,span,depth,world=32,optimizer_per_replica=old['optimizer_matrix_ops_per_replica'])
            if not np.all(work==work[0]) or int(work.sum())>np.iinfo(np.int64).max:raise AssertionError('Constant-depth bounded work ledger required')
            np.save(out/(arm+'-depths.npy'),depth);np.save(out/(arm+'-work.npy'),work)
            if not np.array_equal(np.load(out/(arm+'-depths.npy')),depth) or not np.array_equal(np.load(out/(arm+'-work.npy')),work):raise AssertionError('Fixed table readback differs')
            tables[arm]=dict(depth_file=arm+'-depths.npy',depth_sha256=file_sha256(out/(arm+'-depths.npy')),work_file=arm+'-work.npy',work_sha256=file_sha256(out/(arm+'-work.npy')),matrix_ops_per_update=int(work[0]),forward_steps=n,backward_steps=n)
            work_by_arm[arm]=work
        info=old['initial_states']['s5231-family'];initial_file=parent/info['file']
        if file_sha256(initial_file)!=info['sha256']:raise AssertionError('Original random initializer changed')
        initial=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
        if initial['qualification_only'] is not False or initial['seed']!=5231 or initial['optimizer_updates']!=0 or initial['purpose']!='fresh_random_future_span_scientific_initializer' or initial['config']!=old['model'] or digest(dict(sorted(initial['model'].items())))!=info['model_digest']:raise AssertionError('Only verified original random weights may initialize this curve')
        del initial
        endpoints={};matches={}
        for label,total in [('1b',1920),('2b',3840)]:
            for arm in ['one','fixed']:
                domains=packed.counts(0,total*512);cool=total//2
                endpoints[arm+'-'+label]=dict(arm=arm,total_updates=total,end_targets=sum(domains),domain_targets=domains,end_anchors=int(prefix[total*512]),cooldown_start_update=cool,cooldown_start_targets=sum(packed.counts(0,cool*512)),training_matrix_ops=int(work_by_arm[arm][:total].sum()))
            budget=endpoints['fixed-'+label]['training_matrix_ops'];one_step=tables['one']['matrix_ops_per_update'];total_one=budget//one_step;used=total_one*one_step
            if total_one>=registration['rows'] or not used<=budget<(total_one+1)*one_step:raise AssertionError('Exact integer work endpoint unavailable')
            cool=total_one//2;domains=packed.counts(0,total_one*512)
            endpoints['one-work-'+label]=dict(arm='one',total_updates=total_one,end_targets=sum(domains),domain_targets=domains,end_anchors=int(prefix[total_one*512]),cooldown_start_update=cool,cooldown_start_targets=sum(packed.counts(0,cool*512)),training_matrix_ops=used)
            matches[label]=dict(fixed_budget=budget,one_budget=used,relative_shortfall=(budget-used)/budget,next_update_exceeds=True)
        source_slots=max(x['total_updates'] for x in endpoints.values())*512
        manifest=dict(identity=IDENTITY,registration_sha256=file_sha256(out/'registration.json'),model=asdict(cfg),future_config=asdict(span),data=data,tables=tables,endpoints=endpoints,work_matches=matches,source_windows_needed=source_slots,source_windows_provisioned=registration['rows']*512,initial_state={**info,'file':'../'+parent.name+'/'+info['file']},auxiliary_data=dict(directory=IDENTITY,counts_file='anchor-counts.npy',counts_sha256=file_sha256(out/'anchor-counts.npy'),prefix_file='anchor-prefix.npy',prefix_sha256=file_sha256(out/'anchor-prefix.npy'),source_windows=registration['rows']*512,valid_anchors=int(prefix[-1]),valid_anchors_by_domain=dict(zip(['web','math','code'],map(int,by_domain)))),optimizer_matrix_ops_per_replica=old['optimizer_matrix_ops_per_replica'],routing_seed=old['routing_seed'],original_anchor_prefix_exact=True,scientific_launch_allowed=False,scope=registration['scope'])
        guard()
        if any(file_sha256(f)!=h for f,h in registration['source_sha256'].items()):raise AssertionError('Frozen preparation source changed')
        write_state(out/'manifest.json',manifest);write_state(PUBLIC/'manifest.json',manifest)
        result.update(status='COMPLETE_FIXED_TWO_NTP_MAIN_PREPARATION',manifest_sha256=file_sha256(out/'manifest.json'),work_matches=matches,all_readbacks_exact=True,original_anchor_prefix_exact=True,scientific_launch_allowed=False)
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(wall_seconds=time.time()-started,assigned_chips=0,research_updates=0,test_accessed=False,terminal_alpha_spent=0,scope=registration['scope']);write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':main()
