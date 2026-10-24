"""Qualified scientific trainer with runtime Muon scalar ingestion only."""
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
from scripts import train_low_depth_scientific as trainer


if __name__=='__main__':
    trainer.ParcaePreciseOptimizer=ParcaeRuntimeScalarOptimizer
    trainer.main()
