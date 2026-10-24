"""Prescribed optimizer-only checks; these never constitute model learning."""
import math

import torch


UPDATE_FIXTURE = (
    {'lr': .008, 'momentum': .85, 'weight_decay': .2, 'gradient_scale': 1.},
    {'lr': .0056, 'momentum': .9, 'weight_decay': .16, 'gradient_scale': .75},
    {'lr': .00392, 'momentum': .95, 'weight_decay': .12, 'gradient_scale': -.5},
)


def configure_step(optimizer, update):
    row = UPDATE_FIXTURE[update]
    for group in optimizer.param_groups:
        group['lr'] = row['lr']
        if group['kind'] == 'muon':
            group['momentum'] = row['momentum']
            group['weight_decay'] = row['weight_decay']
    return row['gradient_scale']


def assign_gradient(model, gradients, scale):
    named = dict(model.named_parameters())
    if named.keys() != gradients.keys():
        raise ValueError('Prescribed gradient parameter names changed')
    for name, parameter in named.items():
        value = gradients[name]
        if value.shape != parameter.shape or value.dtype != torch.float32:
            raise ValueError('Prescribed gradient shape/dtype changed')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite prescribed CPU gradient')
        # The reference algorithm modifies its input gradients. Never alias the
        # pinned vector or silently reuse those modified gradients on step two.
        parameter.grad = (value * scale).to(parameter.device).clone()


def named_moments(model, optimizer, *, reference=False):
    """Canonical named CPU states for independent arithmetic comparisons."""
    lookup = {id(p): name for name, p in model.named_parameters()}
    groups = optimizer.param_groups
    result = {}
    for group in groups:
        if group['kind'] == 'adamw':
            state = optimizer.state if reference else optimizer.adamw.state
            for parameter in group['params']:
                item = state[parameter]
                step = item['step']
                result[lookup[id(parameter)]] = {
                    'step': int(step.item() if isinstance(step, torch.Tensor) else step),
                    'm': item['m' if reference else 'exp_avg'].detach().cpu().clone(),
                    'v': item['v' if reference else 'exp_avg_sq'].detach().cpu().clone(),
                }
        else:
            state = optimizer.state if reference else optimizer.muon.state
            item = state[group['params'][0]]
            for index, parameter in enumerate(group['params']):
                result[lookup[id(parameter)]] = {
                    'm': item['m'][index].detach().cpu().clone(),
                    'v': item['v'][index].detach().cpu().clone(),
                }
    return result


def relative_error(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError('Compared tensor names differ')
    error = norm = maximum = 0.
    for name, left in expected.items():
        right = actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Compared tensor shape/dtype differs')
        if not torch.isfinite(right).all() or not torch.isfinite(left).all():
            raise ValueError('Nonfinite comparison input')
        delta = right.double() - left.double()
        error += float(delta.square().sum())
        norm += float(left.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {'relative_l2': math.sqrt(error / max(norm, 1e-30)),
            'maximum_absolute_error': maximum, 'reference_l2': math.sqrt(norm)}


def compare_update(expected_before, expected, actual_before, actual, *, names=None):
    keys = list(expected) if names is None else list(names)
    if any(set(v) != set(expected) for v in [expected_before, actual_before, actual]):
        raise ValueError('Update parameter inventory differs')
    expected_delta = {k: expected[k] - expected_before[k] for k in keys}
    actual_delta = {k: actual[k] - actual_before[k] for k in keys}
    result = relative_error(expected_delta, actual_delta)
    result['scope'] = 'Actual update minus prescribed independent update, normalized by the independent update size.'
    return result


def compare_moments(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError('Optimizer moment names differ')
    left, right = {}, {}
    for name, values in expected.items():
        if values.keys() != actual[name].keys():
            raise ValueError('Optimizer moment fields differ')
        for key, value in values.items():
            other = actual[name][key]
            if isinstance(value, torch.Tensor):
                left[name + ':' + key], right[name + ':' + key] = value, other
            elif value != other:
                raise ValueError('Optimizer scalar step differs')
    return relative_error(left, right)
