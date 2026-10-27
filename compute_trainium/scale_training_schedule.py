"""Frozen token schedules with completed cooldowns; no outcome-dependent gates."""
import math


def rates(exposure,*,warmup_targets,cooldown_targets,end_targets,peak_lr):
    if not (0<exposure<=end_targets and 0<warmup_targets<cooldown_targets<end_targets
            and math.isfinite(peak_lr) and peak_lr>0):raise ValueError('Ordered frozen token milestones required')
    warm=min(1.,exposure/warmup_targets)
    progress=max(0.,(exposure-cooldown_targets)/(end_targets-cooldown_targets))
    return dict(lr=peak_lr*warm*.5*(1+math.cos(math.pi*progress)),
        momentum=.85+.10*warm,weight_decay=.2*(1-progress))


def apply_rates(optimizer,values):
    for group in optimizer.param_groups:
        group['lr']=values['lr']*group.get('base_lr',1.)
        if group['kind']=='muon':
            group['momentum']=values['momentum'];group['weight_decay']=values['weight_decay']
