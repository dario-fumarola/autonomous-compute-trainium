"""Independent pinned-upstream complete scientific gradient for cheap mixing."""
from compute_trainium.mixing_control_reference import construct_reference
from compute_trainium.mixing_control_science import signature
from compute_trainium.parcae_mixing_control import mixing_only_ledger
from compute_trainium import future_span_gradient_point as points
from scripts import audit_future_span_gradient as engine

if __name__=='__main__':
    points.signature=signature
    engine.construct_reference=construct_reference;engine.matmul_ledger=mixing_only_ledger
    engine.main()
