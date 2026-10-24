"""FP32 DEV readout for the separately identified cheap reinjection control."""
import os
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'


def main():
    from compute_trainium.reinjection_control import ReinjectionControlConfig, build_reinjection_control
    from scripts import evaluate_depth_curve as evaluator
    def config(signature):
        if not signature['model'].get('reinjection_only'):
            raise ValueError('Checkpoint is not the trained cheap control')
        return ReinjectionControlConfig(**signature['model'])
    evaluator.config_from_signature = config
    evaluator.build_from_config = build_reinjection_control
    evaluator.main(expected_precision='fp32', allow_test=False)


if __name__ == '__main__':
    main()
