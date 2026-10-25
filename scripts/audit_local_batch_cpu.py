"""Original independent FP32 auditor with an explicit physical-routing check."""
import argparse
import json
from pathlib import Path

from compute_trainium.depth_mixture import window_permutation
from compute_trainium.microbatch_geometry import accumulation_for,physical_slot


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    args,_=parser.parse_known_args();plan=json.loads(args.plan.read_text())
    native=json.loads((args.capture/'report.json').read_text());batch=native['signature']['local_batch']
    key=native['case']['identity'];accumulation=accumulation_for(batch)
    if (plan['local_batches'][key]!=batch or native['signature']['accumulation']!=accumulation
            or native['signature']['global_windows']!=512):raise ValueError('Explicit registered local geometry required')
    for probe in native['probes']:
        for row in probe['rows']:
            hw,sub=physical_slot(row['micro'],batch)
            if (row['hardware_micro'],row['batch_offset'])!=(hw,sub):raise ValueError('Native physical source slot differs')
    def original_audit_routing(cursor,micro,rank,world,old_batch,old_accumulation,seed):
        if (world,old_batch,old_accumulation)!=(32,1,16):raise ValueError('Unexpected reference routing request')
        hw,sub=physical_slot(micro,batch)
        position=(hw*32+rank)*batch+sub
        return [cursor+window_permutation(cursor,512,seed)[position]]
    from scripts import audit_future_span_qualification as engine
    engine.routed_indices=original_audit_routing
    engine.audit()


if __name__=='__main__':main()
