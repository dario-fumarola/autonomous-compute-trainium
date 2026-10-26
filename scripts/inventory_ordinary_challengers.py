"""Shapes and work only: identify conventional controls without training them."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_work_accounting import global_update_work
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    require_inode_reserve([a.root,Path('/')])
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<544*1024**3 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Protected metadata-preparation reserves failed')
    parent_file=a.root/'fixed-two-ntp-main-v1/plan.json';parent=json.loads(parent_file.read_text())
    if any(file_sha256(f)!=sha for f,sha in parent['source_sha256'].items()):raise AssertionError('Reference scientific source changed')
    config=ParcaeCandidateConfig(**parent['model']);target=config.parameter_count(True)
    packing=a.root/parent['data']['packed'];metadata=json.loads((packing/'manifest.json').read_text())
    if (file_sha256(packing/'manifest.json')!=parent['data']['manifest_sha256']
            or file_sha256(packing/'prefix_counts.npy')!=metadata['files']['prefix_counts.npy']):
        raise AssertionError('Original authentic source counts changed')
    counts=np.load(packing/'prefix_counts.npy',mmap_mode='r')
    records=[]
    # These are fixed geometric comparisons,not a quality-selected model sweep.
    choices=[('family-one-6x768',config,True),('ordinary-6x768',config,False),
             ('ordinary-8x640',replace(config,dim=640,heads=5,ffn_dim=2560,core=4),False)]
    for label,c,recurrent in choices:
        with torch.device('meta'):model=ParcaeCandidate(c,recurrent=recurrent,initialize=False)
        parameters=sum(v.numel() for v in model.parameters())
        if parameters!=c.parameter_count(recurrent) or any(v.device.type!='meta' for v in model.parameters()):
            raise AssertionError('Exact meta-device parameter inventory required')
        if not 0<=target-parameters<=.01*target:raise AssertionError('Conventional challenger must fit within1% below the parameter cap')
        work=global_update_work(model,1024,32,1,16,gate_policy='release_muon',forward_steps=1,backward_cap=1)
        tables=(1+sum(c.has_value_embedding(i) for i in range(c.layers)))*c.vocab_size*c.dim
        records.append(dict(label=label,config=asdict(c),recurrent_parameterization=recurrent,
            unique_parameters=parameters,relative_parameters_below_cap=(target-parameters)/target,
            lookup_parameters=tables,other_parameters=parameters-tables,
            unique_layers=c.layers,forward_block_applications=c.layers,head_dim=c.dim//c.heads,
            global_update_work=work,hardware_qualified=False,scientific_fit_admitted=False))
        del model
    lean_fixed_work=326673335904768
    budgets={'completed_full_head_fixed2B':parent['cases']['fixed-2b']['training_dense_matrix_ops'],
             'prospective_lean_fixed2B':3840*lean_fixed_work}
    for row in records:
        per=row['global_update_work']['total_executed_matmuls'];row['prospective_work_controls']={}
        for label,budget in budgets.items():
            updates=budget//per
            if updates*512>=len(counts) or not updates*per<=budget<(updates+1)*per:
                raise AssertionError('Exact provisioned integer source/work floor required')
            row['prospective_work_controls'][label]=dict(target_matrix_ops=budget,updates=int(updates),
                real_targets=int((counts[updates*512]-counts[0]).sum()),actual_matrix_ops=int(updates*per),
                relative_work_shortfall=(budget-updates*per)/budget,next_update_exceeds=True)
    if time.time()-started>180:raise TimeoutError('Metadata-only cap')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='COMPLETE_METADATA_ONLY_ORDINARY_CHALLENGER_INVENTORY',
        source_sha256={f:file_sha256(f) for f in ['scripts/inventory_ordinary_challengers.py',
            'compute_trainium/parcae_candidate.py','compute_trainium/parcae_work_accounting.py']},
        reference_plan_sha256=file_sha256(parent_file),reference_parameter_cap=target,records=records,
        model_forwards=0,optimizer_updates=0,assigned_chips=0,source_examples_read=0,test_accessed=False,
        terminal_alpha_spent=0,wall_seconds=time.time()-started,
        scope='Meta tensors and original TRAIN count metadata only. No random/learned weights,held-out scores,native qualification,tuning or fit. The eight-layer ordinary configuration preserves the existing every-other-layer value-embedding policy and128-dimensional heads;it has four value tables rather than three. Different width/depth/lookup allocation is a prospective conventional challenger,not a parameter-identical paired model or an observed improvement. Work excludes pointwise/AdamW/network/padding and is not chip time. Native numerical checks,independent tuning and a separate frozen budget remain mandatory before any fit. No change to either running queue.')
    write_state(a.output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
