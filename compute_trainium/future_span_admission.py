"""Hardware admission requires completed predecessor work,never an idle gap."""
import json


def predecessor_execution(root):
    experiment=root/'parcae-low-depth-pilot-v1/main-execution-recovery-v2'
    terminal=experiment/'report.json'
    if terminal.exists() and json.loads(terminal.read_text())['status']!='COMPLETE_FROZEN_LOW_DEPTH_MAIN_PILOT':
        raise RuntimeError('Stopped predecessor requires explicit resolution before new native qualification')
    # Private controller writes its main progress to the public aggregate file.
    # The caller supplies its exact repository path separately to avoid relying
    # on an arbitrary process cwd or scanning for a likely progress manifest.
    return experiment


def completed_native_receipts(root,training,gradients):
    expected={f'low-depth-main-r2-{arm}-{phase}' for arm in ['ordinary','fixed1','poisson2'] for phase in ['first','finish','equal']}
    fits=training.get('records',[]);gates=gradients.get('records',[])
    if len(fits)!=9 or {r['identity'] for r in fits}!=expected:return False
    if len(gates)!=9 or {(r['arm'],r['readout']) for r in gates}!={(a,s) for a in ['ordinary','fixed1','poisson2'] for s in ['stable1B','matched','equal']}:return False
    identities=[]
    for row in fits:
        if row['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or row['qualification_only'] or row['test_accessed']:
            raise ValueError('Predecessor native fit receipt differs')
        identities.append(row.get('adopted_from',{}).get('identity',row['identity']))
    for row in gates:
        if row['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT':raise ValueError('Predecessor numerical gate failed')
        identities.append(row['capture_identity'])
    for identity in identities:
        closure=root/'snapshots'/identity/'closure.json'
        if not closure.exists():return False
        if json.loads(closure.read_text())['exit_code']!=0:raise ValueError('Predecessor native closure failed')
    return True
