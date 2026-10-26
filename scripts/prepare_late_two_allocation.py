"""CPU-only exact schedule/work/state preparation; no new science admission."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_scientific_state import matrix_work
from compute_trainium.late_depth_schedule import (matched_late_endpoint,staged_depths,
    switch_update,staged_work,validate_staged_depths)
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_scientific_schedule import source_endpoint
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest

IDENTITY='late-two-allocation-preparation-v1'
PUBLIC=Path('results/late_two_allocation_20261026/preparation')
DOC='docs/LATE_TWO_ALLOCATION_20261026.md'


def read(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/IDENTITY;started=time.time();cpu=time.process_time()
    if out.exists() or PUBLIC.exists(): raise FileExistsError('Preserve preparation attempts')
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3 or time.time()-started>600:
            raise RuntimeError('Bounded preparation reserves/cap')
    guard();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    parent=a.root/'fixed-two-ntp-main-preparation-v1/manifest.json';old=read(parent)
    source_files=['scripts/prepare_late_two_allocation.py','compute_trainium/late_depth_schedule.py',
        'tests/test_late_depth_schedule.py',DOC,'compute_trainium/future_span_scientific_state.py',
        'compute_trainium/parcae_scientific_schedule.py','compute_trainium/parcae_candidate.py']
    registration=dict(identity=IDENTITY,registered_unix=started,parent_manifest_sha256=file_sha256(parent),
        source_sha256={f:file_sha256(f) for f in source_files},seconds_limit=600,model_forwards=0,
        model_updates=0,assigned_chips=0,scientific_launch_allowed=False,scope=Path(DOC).read_text())
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'registration.json',registration);write_state(PUBLIC/'registration.json',registration)
    result=dict(status='STOPPED_LATE_TWO_ALLOCATION_PREPARATION')
    try:
        original_registration=read(parent.parent/'registration.json')
        if any(file_sha256(f)!=h for f,h in original_registration['source_sha256'].items()):
            raise AssertionError('Original qualified fixed-two preparation sources changed')
        cfg=ParcaeCandidateConfig(**old['model']);span=FutureSpanConfig(**old['future_config']);data=old['data']
        packed=a.root/data['packed'];meta=read(packed/'manifest.json')
        if file_sha256(packed/'manifest.json')!=data['manifest_sha256'] or file_sha256(packed/'prefix_counts.npy')!=meta['files']['prefix_counts.npy']:
            raise AssertionError('Accepted source/packing count prefix changed')
        counts=np.load(packed/'prefix_counts.npy',mmap_mode='r')
        if int(counts[-1].sum())!=6500009163: raise AssertionError('Unchanged accepted6.5B unique source required')
        budget=1272689958300549120
        one=old['tables']['one']['matrix_ops_per_update'];two=old['tables']['fixed']['matrix_ops_per_update']
        main=matched_late_endpoint(budget,one,two,maximum_updates=old['source_windows_provisioned']//512)
        auxiliary=old['auxiliary_data'];prefix_path=a.root/auxiliary['directory']/auxiliary['prefix_file']
        if file_sha256(prefix_path)!=auxiliary['prefix_sha256']: raise AssertionError('Original auxiliary counter prefix changed')
        anchors=np.load(prefix_path,mmap_mode='r')
        endpoints={}
        for role,end in [('tune',576),('main',main['total_updates'])]:
            guard();depths=staged_depths(end);validate_staged_depths(depths,end)
            work=matrix_work(cfg,span,depths,world=32,optimizer_per_replica=old['optimizer_matrix_ops_per_replica'])
            if int(work[:end].sum())!=staged_work(end,one,two): raise AssertionError('Independent neural/head/optimizer ledger disagrees')
            switch=switch_update(end);cool=end//2;row=source_endpoint(counts,end,512)
            switched=source_endpoint(counts,switch,512);cooled=source_endpoint(counts,cool,512)
            if end*512>=len(anchors): raise ValueError('Existing exact anchor prefix does not cover proposed unique source')
            if role=='main' and not int(work[:end].sum())<=budget<int(work[:end+1].sum()):
                raise AssertionError('Next fixed-schedule update must exceed original work budget')
            for label,values in [('depths',depths),('work',work)]:
                file=out/(role+'-'+label+'.npy');np.save(file,values)
                if not np.array_equal(np.load(file),values): raise AssertionError('Complete table readback differs')
            row.update(total_updates=end,switch_after_update=switch,switch_targets=switched['real_targets'],
                actual_two_pass_target_fraction=(row['real_targets']-switched['real_targets'])/row['real_targets'],
                cooldown_start_update=cool,cooldown_start_targets=cooled['real_targets'],
                end_anchors=int(anchors[end*512]),training_matrix_ops=int(work[:end].sum()),
                depth_file=role+'-depths.npy',depth_sha256=file_sha256(out/(role+'-depths.npy')),
                work_file=role+'-work.npy',work_sha256=file_sha256(out/(role+'-work.npy')))
            endpoints[role]=row
        initials={}
        for seed in [4127,5231,17293]:
            guard()
            source_plan=a.root/('fixed-two-ntp-tune-v1/plan.json' if seed==4127 else 'fixed-control-replication-main-v1/full-plan.json')
            parent_plan=read(source_plan);info=parent_plan['initial_states'][f's{seed}-family']
            file=(source_plan.parent/info['file']).resolve()
            if not file.is_relative_to(a.root.resolve()) or file_sha256(file)!=info['sha256']:
                raise AssertionError('Exact original random initializer changed')
            saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
            if (saved['qualification_only'] is not False or saved['seed']!=seed or saved['optimizer_updates']!=0
                    or saved['config']!=old['model'] or saved['future_config']!=old['future_config']
                    or digest(dict(sorted(saved['model'].items())))!=info['model_digest']):
                raise AssertionError('Original random model/body/head state differs')
            initials[f's{seed}-family']={**info,'file':str(file.relative_to(a.root)),
                'file_is_root_relative':True,'source_plan_sha256':file_sha256(source_plan)}
            del saved
        fixed=read(a.root/'fixed-two-ntp-main-v1/plan.json');references={}
        for name,file in [('ordinary8','lean-baseline-main-ordinary8-v1/report.json'),
                          ('fixed5231','fixed-two-ntp-main-v1/report.json'),
                          ('fixed17293','fixed-control-replication-main-v1/report.json')]:
            references[name]=dict(file=file,sha256=file_sha256(a.root/file))
        manifest=dict(status='COMPLETE_CPU_ONLY_LATE_TWO_ALLOCATION_PREPARATION',identity=IDENTITY,
            registration_sha256=file_sha256(out/'registration.json'),model=old['model'],future_config=old['future_config'],
            data=data,auxiliary_data=auxiliary,endpoints=endpoints,main_work_floor=main,
            initial_states=initials,training_parameters=166571840,deployed_parameters=141963840,
            optimizer_matrix_ops_per_replica=old['optimizer_matrix_ops_per_replica'],routing_seed=old['routing_seed'],
            source_total_real_targets=6500009163,evaluation=fixed['evaluation'],reference_reports=references,
            original_initializer_bytes_unchanged=True,initializers_copied=False,all_table_readbacks_exact=True,
            scientific_launch_allowed=False,model_forwards=0,model_updates=0,assigned_chips=0,
            test_accessed=False,terminal_alpha_spent=0,scope=registration['scope'])
        guard();write_state(out/'manifest.json',manifest);write_state(PUBLIC/'manifest.json',manifest)
        result.update(status=manifest['status'],manifest_sha256=file_sha256(out/'manifest.json'),endpoints=endpoints,
            original_initializer_bytes_unchanged=True,all_table_readbacks_exact=True)
    except BaseException as error:
        result.update(error=str(error),error_kind=type(error).__name__);raise
    finally:
        result.update(wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            model_forwards=0,model_updates=0,assigned_chips=0,scientific_launch_allowed=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:result[k] for k in ('status','wall_seconds','model_updates')}))


if __name__=='__main__':main()
