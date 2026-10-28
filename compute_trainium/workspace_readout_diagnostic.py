"""Prefix-only readout interventions;not another fitted objective."""
import json
import torch
from .workspace_learning import identity_reader_context
from .workspace_discovery_analysis import clustered_interval

VARIANTS=('sample_prepend','sample_tail','greedy_prepend','greedy_tail')


def readout_context(prefix,workspace,format,*,tail=False,tail_length=16):
    if not tail:return identity_reader_context(prefix,workspace,format)
    if prefix.ndim!=2 or workspace.ndim!=2 or prefix.shape[0]!=1 or tail_length<1:raise ValueError('One prefix and positive replay length required')
    if workspace.shape[1]==0:return prefix.expand(workspace.shape[0],-1)
    expanded=prefix.expand(workspace.shape[0],-1)
    return torch.cat([expanded,workspace,expanded[:,-tail_length:]],1)


def analyze_readouts(root,plan):
    native=root/'runs'/(plan['identity']+'-evaluation');rows={}
    for arm in ['c0','workspace','ordinary','pause','draft']:
        records=sorted([r for rank in range(64) for r in json.loads((native/f'rank{rank}-{arm}-losses.json').read_text())],key=lambda r:r['index'])
        if [r['index'] for r in records]!=list(range(320)):raise AssertionError('Complete original panel required')
        rows[arm]=records
    results={};domains=[r['domain'] for r in rows['ordinary']]
    for variant in VARIANTS:
        comparisons={
            'ordinary_raw_minus_workspace': [('ordinary','raw_first4_nll'),('workspace',variant+'.own_nll')],
            'ordinary_same_readout_minus_workspace': [('ordinary',variant+'.own_nll'),('workspace',variant+'.own_nll')],
            'c0_same_readout_minus_workspace': [('c0',variant+'.own_nll'),('workspace',variant+'.own_nll')],
            'workspace_raw_minus_own': [('workspace','raw_first4_nll'),('workspace',variant+'.own_nll')],
            'workspace_donor_minus_own': [('workspace',variant+'.donor_nll'),('workspace',variant+'.own_nll')],
            'workspace_constant_minus_own': [('workspace',variant+'.constant_nll'),('workspace',variant+'.own_nll')],
            'workspace_tail_only_minus_own': [('workspace','tail_only_nll'),('workspace',variant+'.own_nll')],
            'ordinary_raw_minus_workspace_first_token': [('ordinary','raw_first_token_nll'),('workspace',variant+'.own_first_token_nll')],
        }
        for label,((a,ak),(b,bk)) in comparisons.items():
            values=[ra[ak]-rb[bk] for ra,rb in zip(rows[a],rows[b],strict=True)]
            results[variant+'/'+label]=clustered_interval(values,plan['evaluation_groups'],domains)
    return dict(status='COMPLETE_READOUT_DIAGNOSTIC_ANALYSIS',contrasts=results,
        warning='Outcome-informed fixed2x2 development follow-up on reused TUNE groups.Multiple descriptive contrasts,no multiplicity-adjusted confirmation or new training.No variant is selected using the observed next token.',terminal_alpha_spent=0)
