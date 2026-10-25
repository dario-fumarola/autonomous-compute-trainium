"""Fixed selection and measured conservative budgets for the objective study."""
import math

CELLS=('one-ntp','one-future','recurrent-ntp','recurrent-future')
RATES=(.008,.016,.032)


def select_rates(records):
    chosen={}
    for cell in CELLS:
        rows=[r for r in records if r['cell']==cell]
        if (len(rows)!=3 or {r['peak_lr'] for r in rows}!=set(RATES)
                or any(not math.isfinite(r['metrics']['weighted_nll']) for r in rows)):
            raise ValueError('All three frozen finite tuning opportunities required in every cell')
        winner=min(rows,key=lambda r:(r['metrics']['weighted_nll'],r['peak_lr']))
        chosen[cell]={'peak_lr':winner['peak_lr'],'case_id':winner['case_id'],
            'weighted_nll':winner['metrics']['weighted_nll'],'boundary_winner':winner['peak_lr'] in (RATES[0],RATES[-1])}
    if len(records)!=12:raise ValueError('Unexpected extra tuning outcomes')
    return chosen


def measured_budget(hardware,*,stage):
    """Pessimistic two-half training + explicit CPU/probe/setup allowances.

    A prediction is a feasibility screen,not assigned-time evidence. Actual
    closures and CPU-only tails must replace it in the final resource ledger.
    """
    if hardware['status']!='PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION':
        raise ValueError('Complete independent hardware qualification required')
    if stage not in ('tune','main'):raise ValueError('Frozen study stage required')
    rows=hardware['records']
    if len(rows)!=4 or {r['case']['identity'] for r in rows}!=set(CELLS):raise ValueError('All four measured paths required')
    updates=576 if stage=='tune' else 1920
    repeats=3 if stage=='tune' else 2
    seconds=[];rates={}
    for row in rows:
        cell=row['case']['identity'];warm=row['native']['warm']
        if (warm['warm_updates']!=100 or warm['warm_seconds']<=0
                or not math.isfinite(warm['warm_seconds']) or warm['wall_seconds']<warm['warm_seconds']):
            raise ValueError('Actual100-update training measurement required')
        per_update=warm['warm_seconds']/100
        setup=max(120.,1.5*(warm['wall_seconds']-warm['warm_seconds']))
        segments=1 if stage=='tune' else 2
        # Fifty percent slower than measured warm execution plus conservative
        # cold costs for every separate worker. Never assume mean depth saves
        # physical padding or heterogeneous-worker synchronization cost.
        seconds.extend([updates*per_update*1.5+segments*setup]*repeats)
        rates[cell]={'measured_warm_seconds_per_update':per_update,
            'measured_warm_targets_per_second':warm['warm_targets_per_second'],
            'setup_seconds_allowance_per_worker':setup}
    loads=[0.,0.]
    for duration in sorted(seconds,reverse=True):loads[min(range(2),key=lambda i:loads[i])]+=duration
    # Full512-window trained-gradient checks:four selected TUNE models,or
    # all eight main endpoints. Native and independent CPU costs are both
    # allowed,including their nonoverlapping tails. Readouts also remain paid.
    probes=4 if stage=='tune' else 8
    allowance=probes*(180/2+900/3)+(12 if stage=='tune' else 24)*360/3+300
    predicted=max(loads)+allowance;cap=4*3600 if stage=='tune' else 6*3600
    return {'measurement_by_cell':rates,'training_two_half_seconds_allowance':max(loads),
        'diagnostic_and_readout_seconds_allowance':allowance,'predicted_seconds_with_margin':predicted,
        'execution_seconds_cap':cap,'fits_registered_cap':predicted<=cap,
        'assigned_training_chip_hours_allowance':sum(seconds)*8/3600,
        'whole_instance_opportunity_chip_hours_cap':cap*16/3600,
        'scope':'Prospective feasibility estimate from actual100 warm updates at exact geometry. Not measured scientific throughput or wall-time equivalence. Includes conservative launches,full-gradient diagnostics,CPU scoring/tails;failed work stays in actual receipts.'}
