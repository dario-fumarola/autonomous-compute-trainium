"""Independent CPU-gold comparison of all32 compiled operator replicas."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.repro_sdpa_math2_cpu import comparison


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.time();a.output.mkdir(parents=True,exist_ok=False)
    result={'status':'STOPPED_INDEPENDENT_COMPILED_SDPA_AUDIT'}
    try:
        torch.set_num_threads(4)
        if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen fixture qualification plan changed')
        plan=json.loads(a.plan.read_text());info=plan['fixture_manifest']
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Saved operator sources changed')
        folder=a.root/plan['fixture_directory'];gold_file=folder/info['gold_file']
        if file_sha256(gold_file)!=info['gold_sha256']:raise AssertionError('Independent CPU input-gradient gold changed')
        gold=torch.load(gold_file,map_location='cpu',mmap=True,weights_only=False)['records']
        native=json.loads((a.capture/'report.json').read_text())
        if native['status']!='CAPTURED_COMPILED_SDPA_NUMERICAL_FIXTURES' or native['plan_sha256']!=a.plan_sha:
            raise ValueError('Completed same-plan compiled fixture capture required')
        expected={(case,policy,rank) for case in gold for policy in ['installed_operator','explicit_fp32_attention'] for rank in range(32)}
        actual={(r['fixture'],r['policy'],r['rank']) for r in native['records']}
        if actual!=expected or len(native['records'])!=len(expected):raise ValueError('Every numerical case/policy/device required once')
        groups={};all_workaround=True
        for r in native['records']:
            path=a.capture/r['file']
            if file_sha256(path)!=r['sha256']:raise AssertionError('Compiled fixture packet changed')
            packet=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            values=comparison(gold[r['fixture']],packet);key=(r['fixture'],r['policy'])
            entry=groups.setdefault(key,{'fixture':r['fixture'],'policy':r['policy'],'replicas':0,'worst_relative_l2':{},'passes_registered_gate':True})
            entry['replicas']+=1
            for name,value in values.items():
                old=entry['worst_relative_l2'].get(name)
                if old is None or value['relative_l2']>old['relative_l2']:
                    entry['worst_relative_l2'][name]={**value,'rank':r['rank']}
            passed=values['output']['relative_l2']<=.002 and all(values[n]['relative_l2']<=.02 for n in ['q','k','v'])
            entry['passes_registered_gate'] &= passed
            if r['policy']=='explicit_fp32_attention':all_workaround &= passed
        result={'status':'PASS_COMPILED_SDPA_WORKAROUND_FIXTURES' if all_workaround else 'FAIL_COMPILED_SDPA_WORKAROUND_FIXTURES',
            'records':list(groups.values()),'original_operator_discrepancy_reproduced':any(
                not r['passes_registered_gate'] for r in groups.values() if r['policy']=='installed_operator'),
            'all32_replica_packets_equal':native['all32_replica_packets_equal'],
            'all32_workaround_replicas_pass':all_workaround,'software_fixture_cases':len(gold)}
        if not all_workaround:raise RuntimeError('Compiled workaround failed fixed2% gradient/0.2% output gates')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,research_examples=0,
            test_accessed=False,scope='All32 actual compiled native replicas compared with independently saved CPU PyTorch MATH arrays. Original-path failures are reported,not required to pass. No corpus,optimizer,recovery,speed or learning claim.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
