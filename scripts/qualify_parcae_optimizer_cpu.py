"""CPU update/state checks against the pinned released optimizer algorithm."""
import argparse
import copy
import hashlib
import io
import json
import time
from pathlib import Path

import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.queue import write_state
from scripts.qualify_native import digest
from third_party.parcae_reference import muon_adamw as reference


def unwrapped_reference():
    # Execute its exact tensor algorithm without an unrelated CPU compiler.
    for name in ('adamw_step', 'muon_step'):
        fn = getattr(reference, name)
        setattr(reference, name, getattr(fn, '_torchdynamo_orig_callable', fn))


def compare(*, fused=False, backend=None):
    torch.manual_seed(109)
    c = ParcaeCandidateConfig(dim=64, heads=2, ffn_dim=128, vocab_size=256, max_seq=32)
    model = ParcaeCandidate(c, initialize=False)
    other = copy.deepcopy(model)
    actual = ParcaeOptimizerCandidate(model, gate_policy='release_muon',
                                      fused_adamw=fused, muon_backend=backend)
    groups = reference.get_muon_param_groups(other.named_parameters(), adamw_lr=.008,
        adamw_betas=(.8,.95), adamw_eps=1e-10, adamw_wd=0., muon_lr=.008)
    expected = reference.MuonAdamW(groups)
    lookup = {id(p):name for name,p in other.named_parameters()}
    expected_names = [[lookup[id(p)] for p in g['params']] for g in groups]
    if actual.recipe['groups'] != expected_names:
        raise AssertionError('Release optimizer grouping differs')
    errors = []
    for update in range(5):
        # Include changing rates/momentum and cautious decay, not just first-step
        # initialization. These are numerical fixtures, not research examples.
        for ga, gb in zip(actual.param_groups, groups, strict=True):
            ga['lr'] = gb['lr'] = .008*(.7**update)
            if ga['kind'] == 'muon':
                ga['momentum'] = gb['momentum'] = .85+.02*update
                ga['weight_decay'] = gb['weight_decay'] = .2*(1-update/10)
        generator = torch.Generator().manual_seed(181+update)
        for (_, a), (_, b) in zip(model.named_parameters(), other.named_parameters(), strict=True):
            a.grad = torch.randn(a.shape, generator=generator)*(.01 if update % 2 else .1)
            b.grad = a.grad.clone()
        before = {k:p.detach().clone() for k,p in other.named_parameters()}
        actual.step()
        expected.step()
        difference = norm = maximum = 0.
        for (name,a), (_,b) in zip(model.named_parameters(), other.named_parameters(), strict=True):
            delta = a.detach()-b.detach()
            difference += float(delta.double().square().sum())
            norm += float((b.detach()-before[name]).double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        relative = (difference/max(norm,1e-30))**.5
        if relative > 2e-5 or maximum > 2e-6:
            raise AssertionError(f'Update delta mismatch: relative={relative}, maximum={maximum}')
        state_difference = 0.
        for a,b in zip(model.parameters(), other.parameters(), strict=True):
            if a in actual.muon.state:
                sa, sb = actual.muon.state[a], expected.state[b]
                for key in ('m','v'):
                    state_difference = max(state_difference, float((sa[key]-sb[key]).abs().max()))
        if state_difference != 0:
            raise AssertionError('Muon moments differ from the specified release algorithm')
        errors.append({'update':update+1,'parameter_error_relative_to_reference_update_delta':relative,
            'maximum_absolute_parameter_difference':maximum, 'maximum_muon_moment_difference':state_difference})
    # Complete nested optimizer serialization and a subsequent real step.
    buffer = io.BytesIO()
    torch.save({'model':model.state_dict(),'optimizer':actual.state_dict()},buffer)
    buffer.seek(0)
    saved = torch.load(buffer,weights_only=False)
    restored = ParcaeCandidate(c,initialize=False)
    restored.load_state_dict(saved['model'])
    recovery = ParcaeOptimizerCandidate(restored,gate_policy='release_muon',
                                       fused_adamw=fused,muon_backend=backend)
    recovery.load_state_dict(saved['optimizer'])
    if digest(actual.state_dict()) != digest(recovery.state_dict()):
        raise AssertionError('Complete CPU optimizer state changed on load')
    generator = torch.Generator().manual_seed(901)
    for a,b in zip(model.parameters(),restored.parameters(),strict=True):
        a.grad = torch.randn(a.shape,generator=generator)*.03
        b.grad = a.grad.clone()
    actual.step()
    recovery.step()
    if any(not torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters(),strict=True)):
        raise AssertionError('CPU optimizer trajectory changes after nested-state restoration')
    if digest(actual.state_dict()) != digest(recovery.state_dict()):
        raise AssertionError('Complete CPU optimizer state changed in resumed next step')
    return {'adamw_fused':fused,'muon_backend':backend,'updates':errors,
            'complete_state_roundtrip_and_next_update_exact':True,
            'recipe':actual.recipe}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    started = time.time()
    torch.set_num_threads(2)
    unwrapped_reference()
    files = ['compute_trainium/parcae_optimizer_candidate.py','scripts/qualify_parcae_optimizer_cpu.py',
             'third_party/parcae_reference/muon_adamw.py']
    hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    records = [compare(fused=False),compare(fused=True),compare(fused=True,backend='aot_eager')]
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise AssertionError('Optimizer candidate changed during qualification')
    report = {'status':'PASS_CPU_RELEASE_OPTIMIZER_UPDATE_AND_RECOVERY_CHECKS',
        'source_sha256':hashes,'records':records,'assigned_chips':0,'test_accessed':False,
        'scope':'Five prescribed gradient updates per CPU implementation and exact nested-state restoration/next step. Numerical software fixtures only; no model learning, native arithmetic, distributed recovery or throughput qualification. Gate policy explicitly follows released code.',
        'wall_seconds_after_imports':time.time()-started}
    write_state(a.output/'report.json',report)
    print(json.dumps({'status':report['status'],'records':[{k:r[k] for k in ['adamw_fused','updates','complete_state_roundtrip_and_next_update_exact']} for r in records],'seconds':report['wall_seconds_after_imports']},indent=2))


if __name__ == '__main__':
    main()
