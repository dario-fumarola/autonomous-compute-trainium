"""No-update attention-route control for the registered trained-gradient case.

Disable the SDK's NKI SDPA fast path before importing torch or torch_neuronx.
All model, source, normalization, loss and gradient-capture code is reused.
This diagnostic does not authorize a change to the scientific training recipe.
"""
import os

os.environ["TORCH_NEURONX_ENABLE_NKI_SDPA"] = "0"

from scripts.capture_gradient_precision_control import main


if __name__ == "__main__":
    main()
