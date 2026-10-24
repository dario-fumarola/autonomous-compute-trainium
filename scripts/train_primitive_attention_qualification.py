"""Qualify FP32 primitive-attention updates; never an alternate scientific fit.

Uses the existing packed trainer and its unchanged numerical/recovery guards.
The source precision/topology copy is separately registered and tensor-audited.
"""
import os

os.environ["TORCH_NEURONX_ENABLE_NKI_SDPA"] = "0"

from scripts.train_variable_depth import main


if __name__ == "__main__":
    main()
