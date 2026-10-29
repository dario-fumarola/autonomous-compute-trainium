"""Reuse the qualified graph; annotate its fixed-one scientific scope."""
import json
import os
from pathlib import Path
import sys

from scripts.depth_causal_scientific_entry import main


if __name__ == '__main__':
    original = list(sys.argv)
    entry = original[original.index('--entry')+1]
    main()
    if entry == 'train' and int(os.environ.get('RANK', '0')) == 0:
        output = Path(original[original.index('--output')+1])
        path = output/'report.json'
        report = json.loads(path.read_text())
        report['scope'] = 'One generated latent step on every update, exact depth-causal graph, differentiable lower-stage KV reuse and uniform authentic-token NTP. Only the one-step exit is trained; zero/two exits are not scientific predictions in this experiment.'
        path.write_text(json.dumps(report, indent=2)+'\n')
