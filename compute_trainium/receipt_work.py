"""Known neural matmul work, with explicit accounting for numerical probes."""


def estimate_matmuls(report, *, scalar_training_report=None, dev_documents=None):
    result={'optimizer_update_matmul_flops':0,'known_training_probe_matmul_flops':0,
            'readout_matmul_flops':0,'readout_metadata_reconstructed':False}
    step=report.get('flops_per_global_step',{}).get('training_matmul_flops')
    if step is not None:
        result['optimizer_update_matmul_flops']=report['updates']*step
        # Every rank does one local forward+backward before DDP/updates. The
        # update contains accumulation such microbatches on the same ranks.
        result['known_training_probe_matmul_flops']=step//report['accumulation'] if report.get('cpu_probe') else None
    if 'scored_windows' in report and 'native_probe_windows' in report and 'forward_flops_per_window' in report:
        result['readout_matmul_flops']=(report['scored_windows']+report['native_probe_windows'])*report['forward_flops_per_window']
    elif report.get('targets',0):
        result['readout_matmul_flops']=None
        if (report.get('status')=='PASS_CHECKPOINT_DEV_EVALUATION' and report.get('split')=='dev'
                and scalar_training_report is not None and dev_documents is not None):
            selected=[];cohort=report['cohort'];limit=report['max_documents_per_domain']
            for domain in ['web','math','code']:
                pool=sorted((d for d in dev_documents if d['split']=='dev' and d['domain']==domain
                    and (cohort=='all' or d.get('cohort')==cohort)),key=lambda d:d['id'])
                selected.extend(pool[:limit] if limit else pool)
            if len(selected)!=report['documents'] or sum(len(d['tokens']) for d in selected)!=report['targets']:
                raise ValueError('Historical scalar readout coverage cannot be reconstructed')
            sig=scalar_training_report['signature'];context=sig['context']
            windows=sum((len(d['tokens'])+context-1)//context for d in selected)
            per_window=scalar_training_report['flops_per_global_step']['forward_matmul_flops']//(sig['world']*sig['batch']*sig['accumulation'])
            # Its first scored window is also the CPU probe: no extra native
            # forward call. Do not add the generic sharded evaluator's probe.
            result.update(readout_matmul_flops=windows*per_window,readout_metadata_reconstructed=True)
    return result
