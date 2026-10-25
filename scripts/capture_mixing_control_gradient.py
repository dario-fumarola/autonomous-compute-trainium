"""Zero-update scientific check with the explicit cheap-control identity."""
import os

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate,mixing_only_ledger
    from compute_trainium.mixing_control_science import signature
    from compute_trainium import future_span_gradient_point as points
    from scripts import capture_future_span_gradient as engine
    points.signature=signature
    engine.FutureSpanCandidate=MixingOnlyFutureSpanCandidate;engine.matmul_ledger=mixing_only_ledger
    engine.main()


if __name__=='__main__':main()
