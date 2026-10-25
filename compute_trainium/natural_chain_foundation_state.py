"""Explicit parent selection for the fixed initialization-history comparison."""
from pathlib import PurePosixPath


def foundation_initial(plan,case):
    if (case.get('arm') not in ['one','recurrent'] or case.get('initial') not in ({'one','one_source'} if case.get('arm')=='one' else {'recurrent'})):
        raise ValueError('Case must retain its declared One or recurrent foundation')
    initial=plan['initial_states'][case['initial']]
    if initial.get('role')!=case['arm'] or PurePosixPath(initial['file']).name!=initial['file']:
        raise ValueError('Foundation role or checkpoint basename differs')
    if type(initial.get('parent_cursor')) is not int or initial['parent_cursor']<1:
        raise ValueError('Positive verified foundation source cursor required')
    if plan['ntp_start_cursor']<initial['parent_cursor']:
        raise ValueError('Continuation must be unseen by both foundations')
    return initial
