"""Preserved optimizer capture with explicitly different FP32-polar recipe."""
from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer


def main():
    from scripts import capture_parcae_optimizer_native as capture
    capture.ParcaeOptimizerCandidate=ParcaePreciseOptimizer
    capture.main()


if __name__=='__main__':main()
