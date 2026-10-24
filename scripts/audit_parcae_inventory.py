"""Data-free unique-parameter inventory; embeddings are not dense model FLOPs."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import parameter_groups
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = ParcaeCandidateConfig(vocab_size=32000)
    records = []
    with torch.device('meta'):
        for recurrent in [False, True]:
            model = ParcaeCandidate(config, recurrent=recurrent, initialize=False)
            inventory = Counter()
            for name, parameter in model.named_parameters():
                category = ('tied_input_output' if name == 'transformer.wte.weight' else
                            'lookup_only_value_embeddings' if name.startswith('value_embeds.') else
                            'remaining_transformations_and_gates')
                inventory[category] += parameter.numel()
            total = sum(inventory.values())
            if total != config.parameter_count(recurrent) or model.lm_head.weight is not model.transformer.wte.weight:
                raise AssertionError('Unique count or tied output identity differs')
            groups = parameter_groups(model.named_parameters(), gate_policy='release_muon')
            families = Counter()
            for group in groups:
                families[group['kind']] += sum(p.numel() for p in group['params'])
            if sum(families.values()) != total:
                raise AssertionError('Optimizer parameter inventory differs')
            records.append({'arm': 'recurrent' if recurrent else 'ordinary',
                'unique_parameters': total, 'categories': dict(inventory),
                'fractions': {k: v/total for k, v in inventory.items()},
                'optimizer_parameter_families': dict(families), 'head_tied_to_input': True,
                'recurrent_depth_changes_unique_parameters': False})
    report = {'status': 'COMPLETE_DATA_FREE_PARCAE_PARAMETER_INVENTORY',
        'config': asdict(config), 'records': records,
        'source_sha256': {p: file_sha256(p) for p in ['scripts/audit_parcae_inventory.py',
            'compute_trainium/parcae_candidate.py', 'compute_trainium/parcae_optimizer_candidate.py']},
        'assigned_chips': 0, 'research_optimizer_updates': 0,
        'scope': 'Meta tensors only. Lookup-only tables have gather/scatter costs, not dense projection FLOPs. The tied input table also supplies the dense vocabulary head, whose work must be counted. Recurrent depth-one is the same-parameter family control; plain ordinary omits injection/readout parameters. No source data, learning or performance measurement.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Preserve the prior inventory report')
    write_state(args.output, report)


if __name__ == '__main__':
    main()
