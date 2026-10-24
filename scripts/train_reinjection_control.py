"""Versioned entry for the inexpensive trained boundary-addition control.

Reuse the qualified trainer's routing/optimizer/recovery mechanics. Only its
configuration and model factory are replaced, in this process. The marker in
every checkpoint prevents mistaking this path for an ordinary transformer.
"""
import os
import sys

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'


def main():
    precision = [sys.argv[i+1] for i, v in enumerate(sys.argv[:-1]) if v == '--precision']
    policy = [sys.argv[i+1] for i, v in enumerate(sys.argv[:-1]) if v == '--depth-policy']
    repetitions = [sys.argv[i+1] for i, v in enumerate(sys.argv[:-1]) if v == '--repetitions']
    if (not precision or precision[-1] != 'fp32' or not policy or policy[-1] != 'ordinary'
            or not repetitions or repetitions[-1] != '1'):
        raise ValueError('Cheap control requires explicit FP32 and one physical pass')
    from compute_trainium.reinjection_control import ReinjectionControlConfig, build_reinjection_control
    from scripts import train_variable_depth as trainer
    trainer.PreNormRecurrenceConfig = ReinjectionControlConfig
    trainer.build_from_config = build_reinjection_control
    trainer.main()


if __name__ == '__main__':
    main()
