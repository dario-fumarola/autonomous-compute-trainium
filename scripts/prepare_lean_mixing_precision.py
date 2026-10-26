"""Actual-window CPU parity for the corrected,head-free cheap baseline."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.lean_mixing_precision import LeanFP32MixingCandidate
from compute_trainium.metrics import real_token_losses
from compute_trainium.mixing_precision_probe import PrecisionMixingProbe
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import mixing_only_ledger
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.qualify_native import digest
from scripts.run_mixing_precision_diagnosis import guards

IDENTITY='lean-mixing-precision-preparation-v2'
PUBLIC=Path('results/lean_mixing_precision_20261026/preparation-v2')
DOC='docs/LEAN_MIXING_PRECISION_PREPARATION_20261026.md'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/IDENTITY
    if out.exists():raise FileExistsError('Preserve CPU preparation attempt')
    guards(a.root);torch.set_num_threads(12);torch.set_num_interop_threads(1)
    original_file=a.root/'fixed-two-ntp-main-v1/plan.json';original=json.loads(original_file.read_text())
    diagnosis_dir=a.root/'mixing-precision-diagnosis-v1';diag=json.loads((diagnosis_dir/'report.json').read_text())
    if diag['status']!='COMPLETE_MIXING_PRECISION_DIAGNOSIS' or not diag['comparisons']['mix32']['passes_original_gate']:
        raise ValueError('Completed unchanged-threshold mature precision diagnosis required')
    diagnostic_plan=json.loads((diagnosis_dir/'plan.json').read_text());point=diagnostic_plan['point']
    initial=original['initial_states'][original['cases']['one-2b']['initial_state']]
    initial_file=original_file.parent/initial['file'];mature=a.root/point['checkpoint']
    if file_sha256(initial_file)!=initial['sha256'] or file_sha256(mature)!=point['checkpoint_sha256']:
        raise AssertionError('Unmodified random/mature reference states required')
    sources={**diagnostic_plan['source_sha256']}
    for f in ['compute_trainium/lean_mixing_precision.py','scripts/prepare_lean_mixing_precision.py',
              'tests/test_lean_mixing_precision.py',DOC]:sources[f]=file_sha256(f)
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Qualified source changed')
    c=ParcaeCandidateConfig(**original['model']);span=FutureSpanConfig(**original['future_config'])
    packed=PackedCorpus.from_mapped(a.root/original['data']['packed'],1024,23,original['data']['corpus_sha256'])
    out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=True);rows=[];calls=0;updates=0
    try:
        for label,file,sha in [('random',initial_file,initial['sha256']),('mature',mature,point['checkpoint_sha256'])]:
            guards(a.root)
            if time.time()-started>900:raise TimeoutError('Bounded CPU preparation cap')
            saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
            full=PrecisionMixingProbe(c,span,precision_mode='mix32',initialize=False,fp16_body=False)
            lean=LeanFP32MixingCandidate(c,initialize=False,fp16_body=False)
            full.load_state_dict(saved['model']);lean.load_state_dict(prediction_state(saved['model']))
            start=saved.get('cursor',0);windows=[list(training_window(packed,start+i,c,span,7240)) for i in range(4)]
            if label=='mature':
                capture=a.root/point['failed_capture'];native=json.loads((capture/'report.json').read_text())
                for i,values in enumerate(windows):
                    row=native['rows'][i]
                    if row['window']!=start+i:raise AssertionError('Original exact mature windows required')
                    noise_file=capture/f"h0-rank{row['rank']}.npy"
                    if file_sha256(noise_file)!=native['noise_sha256_by_rank'][row['rank']]:raise AssertionError('Actual native states changed')
                    noise=np.load(noise_file,mmap_mode='r');values[3]=torch.from_numpy(np.array(noise[row['micro']:row['micro']+1],copy=True))
                    if digest(values[3])!=row['h0_digest']:raise AssertionError('Original supplied latent state differs')
            targets=sum(int(v[2].sum()) for v in windows);anchors=sum(int(v[-1].sum()) for v in windows)
            coefficient=objective_coefficients(targets,anchors,0.,1);losses=[0.,0.]
            for values in windows:
                left,_=FutureSpanObjective(full)(*values,coefficient,2,2)
                logits=lean(values[0],values[3],total_steps=2,backward_cap=2)
                right=real_token_losses(logits,values[1],values[2]).sum()*coefficient[0];calls+=2
                if not torch.equal(left.detach(),right.detach()):raise AssertionError('Ordered prediction loss changed')
                losses[0]+=float(left.detach());losses[1]+=float(right.detach());left.backward();right.backward()
            left=dict(full.named_parameters());right=dict(lean.named_parameters())
            if any(not torch.equal(left[k].grad,p.grad) for k,p in right.items()):raise AssertionError('Prediction gradients changed')
            if any(p.grad is None or torch.count_nonzero(p.grad) for k,p in left.items() if k.startswith('future_lm_head.')):
                raise AssertionError('Removed head has nonzero or absent gradient')
            norm_full=torch.nn.utils.clip_grad_norm_(full.parameters(),1.);norm_lean=torch.nn.utils.clip_grad_norm_(lean.parameters(),1.)
            if not torch.isclose(norm_full,norm_lean,rtol=1e-6,atol=1e-7):raise AssertionError('Clipping changed materially')
            optimizers=[ParcaeRuntimeScalarOptimizer(model,gate_policy='release_muon',lr=.001,muon_lr=.001) for model in [full,lean]]
            for optimizer in optimizers:optimizer.step();updates+=1
            maximum=max(float((left[k].detach()-p.detach()).abs().max()) for k,p in right.items())
            for k,p in right.items():torch.testing.assert_close(left[k],p,rtol=1e-6,atol=1e-7)
            if sum(p.numel() for p in lean.parameters())!=141963840:raise AssertionError('Prediction inventory changed')
            neural=512*mixing_only_ledger(c,1024,total_steps=2,backward_cap=2)['training_matmul_flops']
            optimizer_work=32*muon_update_matmuls(lean.named_parameters(),gate_policy='release_muon')
            rows.append(dict(point=label,real_targets=targets,ntp_losses=losses,loss_exact=True,
                prediction_gradients_bit_equal=True,zero_auxiliary_gradients=True,
                clipping_norms=[float(norm_full),float(norm_lean)],maximum_updated_parameter_absolute_error=maximum,
                deployed_parameters=141963840,lean_model_work=neural,replicated_optimizer_work=optimizer_work,
                lean_global_update_work=neural+optimizer_work))
            if file_sha256(file)!=sha:raise AssertionError('Numerical check modified original state')
            del full,lean,saved,left,right,windows,optimizers,optimizer,logits,values;gc.collect()
        report=dict(status='PASS_LEAN_FP32_MIXING_CPU_PARITY_ONLY',records=rows,source_sha256=sources,
            model_forwards=calls,actual_qualification_optimizer_updates=updates,research_optimizer_updates=0,
            assigned_chips=0,test_accessed=False,terminal_alpha_spent=0,native_launch_allowed=False,
            scientific_fit_admitted=False,wall_seconds=time.time()-started,scope=Path(DOC).read_text())
    except BaseException as error:
        report=dict(status='STOPPED_LEAN_MIXING_CPU_PREPARATION',error=str(error),source_sha256=sources,
            model_forwards_completed=calls,actual_qualification_optimizer_updates_completed=updates,
            research_optimizer_updates=0,assigned_chips=0,test_accessed=False,wall_seconds=time.time()-started);raise
    finally:write_state(out/'report.json',report);write_state(PUBLIC/'report.json',report)


if __name__=='__main__':main()
