"""Same qualified model/updates;validate exactly the launcher's copied trees."""
import os

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

from compute_trainium.future_span_primitive_driver import worker_sources


def main():
    from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate
    from compute_trainium.mixing_control_science import matrix_work,signature
    from scripts import train_future_span_scientific as engine
    engine.FutureSpanCandidate=MixingOnlyFutureSpanCandidate
    engine.matrix_work=matrix_work;engine.signature=signature;engine.worker_sources=worker_sources
    engine.main()


if __name__=='__main__':main()
