"""FP32 primitive-attention DEV readout. TEST is deliberately disabled."""
import os

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'

from scripts.evaluate_depth_curve import main


if __name__ == '__main__':
    main(expected_precision='fp32', allow_test=False)
