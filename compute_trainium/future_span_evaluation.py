"""Strictly prefix-available latent-state policy for future objective readouts."""
from .parcae_training_candidate import initial_state


def window_state(config,window,state_seed,length):
    if type(window) is not int or window<0 or type(length) is not int or not 1<=length<=1024:
        raise ValueError('Original nonnegative window position and supported prefix length required')
    # Always draw the full fixed shape,then slice. Neither document identity,
    # future content,target length nor an ordinal sorted by content hashes seeds
    # the predictor. One deterministic exogenous noise policy is shared by cells.
    return initial_state(config,window,state_seed)[:,:length]


def prediction_state(full_state):
    if not all(key in full_state for key in ['future_lm_head.weight','future_lm_head.bias','future_positions']):
        raise ValueError('A complete trained future-head state is required before removing it')
    return {k:v for k,v in full_state.items() if not k.startswith('future_lm_head.') and k!='future_positions'}
