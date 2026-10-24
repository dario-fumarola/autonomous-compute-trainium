"""Capture primitive FP32 batch2 globals for an existing independent reference."""
import os

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'

from scripts.capture_gradient_precision_control import main


if __name__ == '__main__':
    main(batch_choices=(1, 2, 4), accumulation_choices=(2, 4, 8))
