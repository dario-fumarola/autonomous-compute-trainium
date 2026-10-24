"""Independent existing-DEV CPU scoring for the cheap trained control."""
import argparse
import torch


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--checkpoint', required=True)
    args, _ = p.parse_known_args()
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if not saved['signature']['model'].get('reinjection_only'):
        raise ValueError('Explicit reinjection-control checkpoint required')
    del saved
    from compute_trainium.reinjection_control import IndependentReinjectionControl
    from scripts import score_cpu_checkpoint as scorer
    scorer.IndependentModel = IndependentReinjectionControl
    scorer.main()


if __name__ == '__main__':
    main()
