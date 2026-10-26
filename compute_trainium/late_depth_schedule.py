"""Exact integer one-to-two training allocation; no model or launcher change."""
import numpy as np


def switch_update(total_updates):
    if type(total_updates) is not int or total_updates < 4:
        raise ValueError('At least four complete updates are required')
    return 3*total_updates//4


def staged_work(total_updates, one_step, two_step):
    if any(type(v) is not int or v <= 0 for v in (one_step, two_step)) or two_step < one_step:
        raise ValueError('Positive integer work with an actual second-pass cost required')
    early = switch_update(total_updates)
    return early*one_step+(total_updates-early)*two_step


def matched_late_endpoint(budget, one_step, two_step, *, maximum_updates):
    if type(budget) is not int or budget <= 0 or type(maximum_updates) is not int or maximum_updates < 5:
        raise ValueError('Integer budget and provisioned complete source updates required')
    if staged_work(4, one_step, two_step) > budget:
        raise ValueError('Budget cannot fund a staged comparison')
    left, right = 4, maximum_updates
    if staged_work(right, one_step, two_step) <= budget:
        raise ValueError('Provision a next endpoint to prove the work floor')
    while left+1 < right:
        middle = (left+right)//2
        if staged_work(middle, one_step, two_step) <= budget: left = middle
        else: right = middle
    actual = staged_work(left, one_step, two_step)
    return dict(total_updates=left, switch_after_update=switch_update(left),
        training_matrix_ops=actual,budget_matrix_ops=budget,
        unused_budget_matrix_ops=budget-actual,relative_shortfall=(budget-actual)/budget,
        next_endpoint_would_exceed_budget=staged_work(left+1,one_step,two_step)>budget)


def staged_depths(total_updates, *, microbatches=16, world=32):
    early = switch_update(total_updates)
    if min(microbatches, world) < 1: raise ValueError('Positive distributed dimensions required')
    # Include one next-update sentinel for independent cumulative-work audits.
    values = np.ones((total_updates+1,microbatches,world,2),dtype=np.int16)
    values[early:] = 2
    return values


def validate_staged_depths(values, total_updates):
    expected = staged_depths(total_updates,microbatches=values.shape[1],world=values.shape[2])
    if not np.array_equal(values,expected):
        raise ValueError('Only the frozen globally synchronized final-quarter schedule is supported')
    return True
