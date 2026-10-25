"""Independent consumed-DEV scoring;future head removed,one actual core call."""
from compute_trainium.mixing_control_science import make_reference,signature
from scripts import evaluate_future_span_cpu as engine

if __name__=='__main__':
    engine.make_reference=make_reference;engine.signature=signature;engine.main()
