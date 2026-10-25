"""Resolve the auxiliary body-gradient signal at the shared qualification initializer.

No new forwards,backwards,device work,training or held-out access. Differencing
FP32 packets can amplify rounding,so this is diagnostic rather than a new gate.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def metrics(expected_ntp,expected_future,native_ntp,native_future,names):
    base=signal=actual=error=product=alignment=0.;maximum=0.
    for name in names:
        base_value=expected_ntp[name].double()
        wanted=expected_future[name].double()-base_value
        observed=native_future[name].double()-native_ntp[name].double()
        difference=observed-wanted
        base+=float(base_value.square().sum());signal+=float(wanted.square().sum())
        actual+=float(observed.square().sum());error+=float(difference.square().sum())
        product+=float((wanted*observed).sum());alignment+=float((wanted*base_value).sum())
        maximum=max(maximum,float(difference.abs().max()))
    return {'ntp_l2':base**.5,'cpu_auxiliary_difference_l2':signal**.5,
        'native_auxiliary_difference_l2':actual**.5,
        'cpu_auxiliary_relative_to_ntp':(signal/max(base,1e-60))**.5,
        'native_vs_cpu_auxiliary_difference_relative_l2':(error/max(signal,1e-60))**.5,
        'auxiliary_direction_cosine':product/max((signal*actual)**.5,1e-30),
        'cpu_auxiliary_vs_ntp_cosine':alignment/max((signal*base)**.5,1e-30),
        'maximum_absolute_difference':maximum}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    started=time.time();torch.set_num_threads(2)
    qualification=a.root/'future-span-math-qualification-v2'
    complete=json.loads((qualification/'report.json').read_text())
    if complete['status']!='PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION':raise ValueError('Complete frozen qualification required')
    public=Path('results/future_span_initial_signal_20261025');public.mkdir(exist_ok=False)
    registration={'registered_unix':started,'source_sha256':file_sha256(__file__),
        'qualification_report_sha256':file_sha256(qualification/'report.json'),
        'comparisons':['one:initial future-minus-ntp','recurrent:initial future-minus-ntp'],
        'scope':'Post-qualification numerical diagnostic at shared qualification-only initializer. No new research examples,quality score or criterion;no live recipe changes.'}
    write_state(public/'plan.json',registration);rows=[]
    for arm in ['one','recurrent']:
        packets={};provenance={};fingerprints=[]
        for objective in ['ntp','future']:
            key=arm+'-'+objective;record=next(r for r in complete['records'] if r['case']['identity']==key)
            cpu=record['initial_cpu'];fingerprints.append(cpu['gold_fingerprint'])
            folder=qualification/'cpu'/key/'warm-u0';gold=folder/'cpu-global.pt'
            capture=a.root/'runs'/('future-span-math-qualification-v2-'+key+'-warm')
            native=record['native']['warm'];point=next(p for p in native['probes'] if p['update']==0)
            vector=capture/point['gradient_file']
            if file_sha256(gold)!=cpu['cpu_global_sha256'] or file_sha256(vector)!=point['gradient_sha256']:
                raise AssertionError('Original full-objective gradient packet changed')
            packets['cpu-'+objective]=torch.load(gold,map_location='cpu',mmap=True,weights_only=False)['gradients']
            packets['native-'+objective]=torch.load(vector,map_location='cpu',mmap=True,weights_only=False)['gradients']
            provenance[objective]={'independent_cpu_sha256':file_sha256(gold),'native_sha256':file_sha256(vector)}
        fields=['model_digest','source_cursor','real_targets','valid_anchors','noise_sha256_by_rank','depth_table_sha256','plan_sha256']
        if any(fingerprints[0][k]!=fingerprints[1][k] for k in fields):raise AssertionError('Common weights/source/depth/noise required')
        names=list(packets['cpu-ntp'])
        if any(set(value)!=set(names) for value in packets.values()):raise AssertionError('Complete matching parameter inventory required')
        args=[packets[k] for k in ['cpu-ntp','cpu-future','native-ntp','native-future']]
        rows.append({'arm':arm,'shared_initial_model_digest':fingerprints[0]['model_digest'],
            'provenance':provenance,'backbone':metrics(*args,[n for n in names if not n.startswith('future_lm_head.')]),
            'auxiliary_head':metrics(*args,[n for n in names if n.startswith('future_lm_head.')])})
        del packets,args
    result={'status':'COMPLETE_INITIAL_AUXILIARY_SIGNAL_DIAGNOSTIC','records':rows,
        'wall_seconds':time.time()-started,'optimizer_updates':0,'assigned_chips':0,'test_accessed':False,
        'scope':'Differences of separately accumulated FP32 gradient packets at the identical qualification initializer and full512 original windows. Quantifies numerical resolution of the auxiliary signal,not a pure-auxiliary backward,training-trajectory explanation,scientific effect or changed admission gate.'}
    write_state(public/'report.json',result)
    print(json.dumps({'status':result['status'],'records':rows,'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
