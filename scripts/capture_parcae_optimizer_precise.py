"""Preserved optimizer capture with explicitly different FP32-polar recipe."""
import os

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer


def eager_fp32_optimizer(model,*,gate_policy,fused_adamw,muon_backend):
    if muon_backend!='neuron':raise ValueError('Unexpected preserved capture execution request')
    return ParcaePreciseOptimizer(model,gate_policy=gate_policy,
                                 fused_adamw=fused_adamw,muon_backend=None)


def main():
    from scripts import capture_parcae_optimizer_native as capture
    capture.ParcaeOptimizerCandidate=eager_fp32_optimizer
    capture.main()


if __name__=='__main__':main()
