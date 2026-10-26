"""Prospective DEV gate for spending on lean numerical qualification."""
import math


def passes_development_gate(report):
    if report.get('status')!='COMPLETE_FIXED_CONTROL_DEVELOPMENT_REPLICATION':
        raise ValueError('Only the complete five-endpoint development study is eligible')
    if (report.get('test_accessed') is not False or report.get('terminal_alpha_spent')!=0
            or report.get('terminal_success') is not False):
        raise ValueError('This is a development resource gate, not terminal inference')
    comparisons=report['comparisons'];required={f'{s}-{k}' for s in [5231,17293]
                                               for k in ['equal_data','one_work','cheap_work']}
    if set(comparisons)!=required:
        raise ValueError('Both seeds and both work controls must be retained')
    gains={key:float(value['gain_nats']) for key,value in comparisons.items()}
    if not all(math.isfinite(v) for v in gains.values()):raise ValueError('Finite completed contrasts required')
    return all(all(gains[f'{s}-{k}']>0 for s in [5231,17293])
               and sum(gains[f'{s}-{k}'] for s in [5231,17293])/2>=.01
               for k in ['one_work','cheap_work'])
