"""Explicit FP32/primitive-SDPA entry for the registered development execution."""
import os
import sys

precisions = [sys.argv[i+1] for i, value in enumerate(sys.argv[:-1]) if value == '--precision']
if not precisions or precisions[-1] != 'fp32':
    raise ValueError('This entry requires an explicit final --precision fp32')
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'

from scripts.train_variable_depth import main


if __name__ == '__main__':
    main()
