"""Preserved capture with eager Muon: isolate compile-induced BF16 differences."""
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate


def eager_optimizer(model, *, gate_policy, fused_adamw, muon_backend):
    if muon_backend != 'neuron':
        raise ValueError('This wrapper replaces only the registered native Muon compilation')
    return ParcaeOptimizerCandidate(model, gate_policy=gate_policy,
                                    fused_adamw=fused_adamw, muon_backend=None)


def main():
    from scripts import capture_parcae_optimizer_native as capture
    capture.ParcaeOptimizerCandidate = eager_optimizer
    capture.main()


if __name__=='__main__':main()
