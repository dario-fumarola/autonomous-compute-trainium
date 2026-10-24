"""Whole-trainer replay checks in units of an update, with named moments."""
import torch

from .parcae_update_qualification import compare_moments, compare_update


def moments(state):
    recipe=state['recipe']['groups']
    result={}
    adam=state['adamw']
    if len(adam['param_groups'])!=1 or len(recipe[0])!=len(adam['param_groups'][0]['params']):
        raise ValueError('AdamW recipe/state inventory differs')
    for name,index in zip(recipe[0],adam['param_groups'][0]['params'],strict=True):
        item=adam['state'][index]
        result[name]={'step':int(item['step']),'m':item['exp_avg'],'v':item['exp_avg_sq']}
    muon=state['muon']
    for names,group in zip(recipe[1:],muon['param_groups'],strict=True):
        if len(names)!=len(group['params']):raise ValueError('Muon recipe/state inventory differs')
        item=muon['state'][group['params'][0]]
        if item['m'].shape[0]!=len(names) or item['v'].shape[0]!=len(names):
            raise ValueError('Muon state leading dimension differs')
        for i,name in enumerate(names):
            result[name]={'m':item['m'][i],'v':item['v'][i]}
    return result


def check_replay(before,uninterrupted,replayed):
    recipe=before['optimizer']['recipe']
    if uninterrupted['optimizer']['recipe']!=recipe or replayed['optimizer']['recipe']!=recipe:
        raise ValueError('Replay optimizer recipe differs')
    names=[n for group in recipe['groups'] for n in group]
    if len(set(names))!=len(names):raise ValueError('Duplicated named optimizer parameters')
    for state in [before,uninterrupted,replayed]:
        if not torch.equal(state['model']['lm_head.weight'],state['model']['transformer.wte.weight']):
            raise ValueError('Tied head values differ')
    buffers=set(before['model'])-set(names)-{'lm_head.weight'}
    if set(uninterrupted['model'])!=set(before['model']) or set(replayed['model'])!=set(before['model']):
        raise ValueError('Complete model inventory differs')
    if any(not torch.equal(before['model'][n],s['model'][n]) for n in buffers for s in [uninterrupted,replayed]):
        raise ValueError('Persistent model buffer changed')
    initial={n:before['model'][n] for n in names}
    left={n:uninterrupted['model'][n] for n in names}
    right={n:replayed['model'][n] for n in names}
    lm,rm=moments(uninterrupted['optimizer']),moments(replayed['optimizer'])
    result={'update':compare_update(initial,left,initial,right), 'moments':compare_moments(lm,rm)}
    families={}
    for label,selected in [('adamw',recipe['groups'][0]),('muon',[n for g in recipe['groups'][1:] for n in g])]:
        families[label]={'update':compare_update(initial,left,initial,right,names=selected),
            'moments':compare_moments({n:lm[n] for n in selected},{n:rm[n] for n in selected})}
    result['families']=families
    result['passes_two_percent_gate']=all(v[k]['relative_l2']<=.02 for v in [result,*families.values()] for k in ['update','moments'])
    return result
