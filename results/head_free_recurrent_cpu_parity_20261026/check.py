"""Bounded algebra/software parity only;numeric fixtures are not research data."""
import gc
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback

START=time.time()
import torch
from torch.nn import functional as F
from compute_trainium.future_span import FutureSpanConfig,future_span_targets
from compute_trainium.future_span_sparse import sparse_targets,sparse_future_loss_sum
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.qualify_native import digest

torch.set_num_threads(4);torch.set_num_interop_threads(1)
BASE=Path(__file__).parent
PLAN=json.loads((BASE/'registration.json').read_text())
ROOT=Path(PLAN['root'])
OUT=BASE/'report.json'
if OUT.exists():raise FileExistsError('Preserve closed CPU parity attempt')
RESULT=dict(status='STOPPED_HEAD_FREE_RECURRENT_CPU_PARITY',model_forwards=0,optimizer_updates=0,assigned_chips=0,records=[],active_main_sources_verified_initially=False,active_main_sources_verified_finally=False)
def guard():
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if (time.time()-START>PLAN['seconds_limit'] or available<544*1024**3
            or min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/').free)<35*1024**3):
        raise RuntimeError('Original CPU parity time/memory/storage cap')
def interrupted(number,frame):raise TimeoutError('Bounded CPU parity interrupted')
for number in [signal.SIGINT,signal.SIGTERM]:signal.signal(number,interrupted)

try:
    guard()
    assert file_sha256(__file__)==PLAN['script_sha256']
    active_file=ROOT/PLAN['active_main_plan'];active=json.loads(active_file.read_text())
    assert file_sha256(active_file)==PLAN['active_main_plan_sha256']
    assert all(file_sha256(f)==h for f,h in active['source_sha256'].items())
    RESULT['active_main_sources_verified_initially']=True
    cfg=ParcaeCandidateConfig(**PLAN['model']);span=FutureSpanConfig(**PLAN['future_config'])
    # Hand-coded token IDs/h0 exercise tensor algebra only. They never enter
    # a training optimizer or any research quality/scientific evaluation set.
    source=((torch.arange(span.context+1)*17+91)%cfg.vocab_size)[None,:]
    tokens=source[:,:-1];target=source[:,1:]
    h0=torch.randn((1,span.context,cfg.dim),generator=torch.Generator().manual_seed(721))
    labels=sparse_targets(future_span_targets(source,torch.ones_like(source,dtype=torch.bool),span,cfg.vocab_size))
    for label,point in PLAN['points'].items():
        guard();file=ROOT/point['file'];assert file_sha256(file)==point['sha256']
        saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
        state=saved['model']
        for fp16 in [False,True]:
            guard();torch.manual_seed(563)
            paid=FutureSpanMathSDPACandidate(cfg,span,recurrent=True,fp16_body=fp16).train()
            lean=ParcaePrimitiveSDPACandidate(cfg,recurrent=True,fp16_body=fp16).train()
            paid.load_state_dict(state,strict=True)
            lean.load_state_dict({k:v for k,v in state.items() if not k.startswith('future_lm_head.') and k!='future_positions'},strict=True)
            assert sum(p.numel() for p in paid.parameters())==166571840
            assert sum(p.numel() for p in lean.parameters())==141963840
            before_paid=digest(dict(sorted(paid.state_dict().items())))
            before_lean=digest(dict(sorted(lean.state_dict().items())))
            RESULT['model_forwards']+=1
            logits,aux=paid(tokens,h0.clone(),total_steps=2,backward_cap=2)
            paid_logits=logits.detach().clone()
            ntp=F.cross_entropy(logits.flatten(0,1),target.flatten())
            future=sparse_future_loss_sum(aux,labels.token_ids,labels.distinct,labels.valid,span.horizon)/labels.valid.sum()
            loss=ntp+future*0.
            paid_ntp=float(ntp.detach());loss.backward()
            del logits,aux,ntp,future,loss
            guard();RESULT['model_forwards']+=1
            lean_logits=lean(tokens,h0.clone(),total_steps=2,backward_cap=2)
            maximum_logits_error=float((paid_logits-lean_logits.detach()).abs().max())
            lean_ntp=F.cross_entropy(lean_logits.flatten(0,1),target.flatten())
            ntp_difference=float(lean_ntp.detach())-paid_ntp
            lean_ntp.backward();del lean_logits,lean_ntp,paid_logits
            paid_named=dict(paid.named_parameters());lean_named=dict(lean.named_parameters())
            assert set(lean_named)=={k for k in paid_named if not k.startswith('future_lm_head.')}
            numerator=denominator=0.;maximum_gradient_error=0.;all_gradients_equal=True;worst=0.
            for key,p in lean_named.items():
                left=paid_named[key].grad;right=p.grad
                assert left is not None and right is not None
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                all_gradients_equal=all_gradients_equal and torch.equal(left,right)
                difference=left.double()-right.double()
                error=float(difference.square().sum());scale=float(left.double().square().sum())
                numerator+=error;denominator+=scale
                maximum_gradient_error=max(maximum_gradient_error,float(difference.abs().max()))
                worst=max(worst,math.sqrt(error/max(scale,1e-30)))
            zero_head=True
            for key,p in paid_named.items():
                if key.startswith('future_lm_head.'):
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    zero_head=zero_head and not bool(torch.count_nonzero(p.grad))
            relative=math.sqrt(numerator/max(denominator,1e-30))
            assert denominator>0 and zero_head
            assert maximum_logits_error<=2e-6 and abs(ntp_difference)<=2e-6
            assert relative<=1e-6 and worst<=1e-6
            assert digest(dict(sorted(paid.state_dict().items())))==before_paid
            assert digest(dict(sorted(lean.state_dict().items())))==before_lean
            row=dict(point=label,cpu_fp16_body=fp16,executed_depth=2,backward_depth=2,
                maximum_logit_absolute_difference=maximum_logits_error,ntp_loss_difference=ntp_difference,
                body_gradient_relative_l2=relative,maximum_parameter_gradient_relative_l2=worst,
                maximum_gradient_absolute_difference=maximum_gradient_error,all_body_gradients_bitwise_equal=all_gradients_equal,
                future_head_gradient_exactly_zero=zero_head,model_states_unchanged=True,checkpoint_sha256=point['sha256'])
            RESULT['records'].append(row);write_state(BASE/'progress.json',dict(records=RESULT['records'],wall_seconds=time.time()-START))
            del paid,lean,paid_named,lean_named;gc.collect()
        del saved,state;gc.collect()
    guard();assert RESULT['model_forwards']==8 and len(RESULT['records'])==4
    assert all(file_sha256(f)==h for f,h in active['source_sha256'].items())
    RESULT['active_main_sources_verified_finally']=True
    RESULT['status']='PASS_HEAD_FREE_RECURRENT_CPU_PREDICTION_AND_GRADIENT_PARITY'
except BaseException as error:
    RESULT.update(error=str(error),error_kind=type(error).__name__,traceback=traceback.format_exc())
    raise
finally:
    RESULT.update(registration_sha256=file_sha256(BASE/'registration.json'),script_sha256=PLAN['script_sha256'],
        active_main_unchanged=RESULT['active_main_sources_verified_initially'] and RESULT['active_main_sources_verified_finally'],wall_seconds=time.time()-START,cpu_seconds=time.process_time(),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        test_accessed=False,terminal_alpha_spent=0,
        scope='CPU algebra/parity screen on generated numerical correctness tensors at original random and mature weights. These tensors are not research training/evaluation examples. Both full-gradient two-pass paths use the original source/configuration with only the unused head removed. No optimizer step,natural-panel score,Native qualification,hardware saving or scientific advantage is established.')
    write_state(OUT,RESULT);print(json.dumps(RESULT),flush=True)
