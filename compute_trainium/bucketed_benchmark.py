"""Whole-update ABBA aggregation with explicit pairing and no promotion."""
import math


ORDER=['original1','bucketed1','bucketed2','original2']


def summarize(trials):
    if [r['trial'] for r in trials]!=ORDER:raise ValueError('Complete original/bucketed/bucketed/original order required')
    means={}
    for row in trials:
        warm=[r for r in row['updates'] if 4<=r['update']<=103]
        if [r['update'] for r in warm]!=list(range(4,104)):raise ValueError('Exactly100 ordered whole warm updates required')
        if any(not math.isfinite(r['seconds']) or r['seconds']<=0 for r in warm):raise ValueError('Finite positive durations required')
        means[row['trial']]=sum(r['seconds'] for r in warm)/100
    ratios=[means['bucketed1']/means['original1'],means['bucketed2']/means['original2']]
    return dict(mean_warm_seconds=means,order_specific_ratios=ratios,
        speed_gate_passed=all(r<=.95 for r in ratios),automatic_promotion=False)
