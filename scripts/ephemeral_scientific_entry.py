"""Scientific entry for a separately admitted, qualified ephemeral control."""
import argparse
import runpy
import sys

from scripts.ephemeral_entry import install


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry', required=True,
                        choices=['train', 'scientific-audit', 'evaluate'])
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    install()
    module = {'train': 'scripts.train_interleaved_screen',
              'scientific-audit': 'scripts.audit_interleaved_screen',
              'evaluate': 'scripts.evaluate_interleaved_screen'}[args.entry]
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
