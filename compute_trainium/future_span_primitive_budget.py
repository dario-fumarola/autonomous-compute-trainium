"""Prospective cap amendment for the measured full-gradient primitive path."""
from .future_span_study_plan import measured_budget as original_measured_budget


def measured_budget(hardware,*,stage):
    result=original_measured_budget(hardware,stage=stage)
    if stage=='main':
        result={**result,'supersedes_unexecuted_main_cap_seconds':result['execution_seconds_cap'],
            'execution_seconds_cap':8*3600,'fits_registered_cap':result['predicted_seconds_with_margin']<=8*3600,
            'whole_instance_opportunity_chip_hours_cap':8*16,
            'prospective_amendment':'Before any objective scientific scores,retain the frozen1B exposure and two paired pilot seeds;allow8h for the measured full-gradient primitive-attention path including cold setup,complete-gradient audits and scoring. TUNE remains4h. No automatic extension after admission.'}
    return result
