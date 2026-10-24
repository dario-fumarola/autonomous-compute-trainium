import torch

from compute_trainium.parcae_optimizer_candidate import muon_tensor_step
from scripts.repro_muon_polar_precision import errors, polar_trace


def test_polar_trace_matches_release_variance_and_diagnoses_dtype():
    for shape in [(1,12,8),(1,8,12)]:
        gradient=torch.randn(shape,generator=torch.Generator().manual_seed(91))*.03
        trace=polar_trace(gradient)
        rd=-1 if shape[-2]>=shape[-1] else -2
        mean=trace[-1].float().square().mean(rd,keepdim=True)
        m=torch.zeros_like(gradient);v=torch.zeros_like(mean);parameters=torch.randn_like(gradient)
        scalars=[torch.tensor(x) for x in [.85,.008,.2,.95]]
        muon_tensor_step(gradient.clone(),parameters,m,v,*scalars,5,rd)
        assert torch.equal(m,trace[0])
        beta=scalars[-1].bfloat16()
        assert torch.equal(v,mean*(1-beta).float())
        assert all(row['relative_l2']==0 for row in errors(trace,trace))
        assert all(x.dtype==torch.float32 for x in polar_trace(gradient,polar_dtype=torch.float32))
