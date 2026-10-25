"""Same independent scoring engine,with correct read-only depth convention."""
from compute_trainium.mixing_prediction_reference_v2 import make_reference
from compute_trainium.mixing_control_science import signature
from scripts import evaluate_future_span_cpu as engine

if __name__=='__main__':
    engine.make_reference=make_reference;engine.signature=signature;engine.main()
