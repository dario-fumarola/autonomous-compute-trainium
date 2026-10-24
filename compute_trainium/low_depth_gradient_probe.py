"""Pure identity checks for zero-update mature-gradient probes."""
from .parcae_scientific_state import signature


def validate_probe_state(saved,parent_sha,parent,case_id):
    case=parent['cases'][case_id]
    qualification=parent['purpose']=='low_depth_driver_integration_qualification'
    if (saved['purpose']!=parent['purpose'] or saved['qualification_only'] is not qualification
        or saved['case_id']!=case_id or saved['signature']!=signature(parent_sha,parent,case)):
        raise ValueError('Probe must use the exact registered model/data/case checkpoint')
    windows=32*parent['accumulation']
    if saved['cursor']!=saved['update']*windows or not 0<saved['update']<=case['total_updates']:
        raise ValueError('Probe checkpoint cursor/update differs')
    return case
