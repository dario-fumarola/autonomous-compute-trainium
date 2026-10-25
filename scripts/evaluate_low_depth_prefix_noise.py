"""Same frozen CPU prediction evaluator with prefix-position-only randomness."""
import json
import sys
from pathlib import Path

from compute_trainium.prefix_noise_policy import prefix_window_state_key
from compute_trainium.training import file_sha256
from scripts import evaluate_low_depth_cpu as evaluator


def main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original = evaluator.window_state_key
    try:
        evaluator.window_state_key = prefix_window_state_key
        evaluator.main()
    finally:
        evaluator.window_state_key = original
    report_file = output / 'report.json'
    report = json.loads(report_file.read_text())
    report.update(status='COMPLETE_PREFIX_ONLY_RANDOMNESS_AUDIT_SCORE',
                  noise_policy='1000000007 + 1024 * original window index; document identity ignored',
                  policy_source_sha256=file_sha256('compute_trainium/prefix_noise_policy.py'),
                  frozen_evaluator_sha256=file_sha256(evaluator.__file__),
                  scope=report['scope'] + ' Separate prospective randomness-policy audit; '
                  'does not replace the frozen main readout or retune any model.')
    report_file.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
